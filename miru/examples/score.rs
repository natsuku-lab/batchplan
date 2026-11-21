//! Scores looks against the simulator's truth by the rules of batchplan-orchestrator's perception
//! scorecard (`bpo-bench scorecard`: `scorecard.py` on its `exp/depth` branch, at 1e7d258), so
//! that what `miru::look` makes of a recorded look and what the executor that recorded it measured
//! are quoted under one definition.
//!
//!   cargo run --release --example score -- <set> [--recorded | --masks truth] [--again]
//!       [--noise <factor>] [--shares] [--offsets] [--twin <set>] [--threads <n>]
//!       [--images <dir>] [--digest] [run ...]
//!
//! `<set>` holds a folder per run, each with its fixtures and a folder per look
//! (`examples/common`). `--recorded` scores the objects the recording's executor names under each
//! pixel in place of miru's. `--masks truth` gives miru the simulator's truth as every view's
//! masks, each pixel numbered by the truth object it shows: what no mask model can better, so
//! what is still wrong with it is the look's own to mend. `--again` gives miru each look's frames
//! a second time, after the scene they gave, and says how many of that scene's objects keep their
//! ids, each held against the object in its place in the second scene, and how many stay unseen:
//! with nothing changed, all should keep them and none stay. `--noise` multiplies every view's
//! stated noise by a factor, so 0.25 says each camera reads four times as exactly as its
//! recording states. `--twin` looks at the same runs of a
//! second set too, the same scenes on other depth (the simulator's exact depth, where `<set>` is a
//! camera's), and holds each truth object matched in both against itself, per kind: how much the
//! shape of `<set>` takes up beyond the twin's, from above (the mean half width over the
//! directions there, and the longest half side) and in height (the half height), and how many are
//! round and solid where the twin is; the twin's views keep the noise their recording states.
//! Masks, a second look, the noise, a twin and a digest (`--digest`) are miru's alone, and
//! refused with `--recorded`.
//! `--threads` is how many threads a look may use, the machine's own number by default.
//! `--shares` adds, per camera and kind of truth object, the share of its pixels that are an
//! object's, a known shape's and nothing's, and the same of the pixels that show no truth object:
//! what of a thing was left to the table, and what of the table was taken for a thing.
//! `--offsets` adds, per kind of truth object, the median of how far a matched one's measured
//! middle lies from its origin seen from above, over a look's cameras together: the scorecard's
//! median offset. A measured middle is that of an object's shape, or with `--recorded` the
//! `center` the executor gave. Beside it, how many of the matched objects are solid
//! (`miru::Object::solid`), which the executor does not say.
//! `--images` writes each view's labels as a PNG: an object in a colour of its own, known shapes
//! dark, the robot light, nothing black. `--digest` adds one hash over every scene miru's looks
//! gave, the `--again` ones among them, and over nothing else: two builds that give the same scenes
//! print the same digest, whatever their times. Naming runs scores those alone.
//!
//! Only pixels with a reading count. A measured object is *of* a truth object when more than half
//! of its pixels on truth objects show that one, and a truth object *lies in* a measured object
//! when more than half of its own pixels are that one's. A truth object that 50 pixels or more
//! show comes out as the first of these that holds, in each camera and over a look's cameras
//! together (their pixels added up, a measured object being one across them):
//!
//! - **masked**: most of it is under the robot's mask;
//! - **merged**: it lies in a measured object that is not of it, or a measured object of it has
//!   another truth object of 50 pixels or more lying in it;
//! - **pieces**: several measured objects are of it, however small;
//! - **matched**: one measured object is of it, and it lies in that one;
//! - **missed**: none of these.
//!
//! Beside each stand two shares of the measured object that holds most of it: `held`, the share
//! of the truth object's pixels in it, and `pure`, the share of its pixels on truth objects that
//! are this one's.

mod common;

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use batchplan::{DepthImage, Device, Obstacle};
use common::{Camera, Error, Recorded};
use glam::Vec2;
use miru::{Label, Look, LookOptions, Noise, Scene, Shape, Support, View};

/// A truth object shown by fewer pixels is not in view: it is not scored, and lies in nothing.
const IN_VIEW: usize = 50;

/// The verdicts, in the order of a tally's columns.
const VERDICTS: [&str; 5] = ["matched", "merged", "pieces", "masked", "missed"];

/// Per kind of truth object, how many came out as each of `VERDICTS`.
type Tally = BTreeMap<String, [usize; 5]>;

/// Per kind of truth object, and under [`NO_OBJECT`] for what shows none: how many of its pixels
/// are an object's, a known shape's and nothing's.
type Shares = BTreeMap<String, [usize; 3]>;

/// What the pixels that show no truth object are counted under.
const NO_OBJECT: &str = "(no truth object)";

/// What the command line asks for.
struct Asked {
    set: PathBuf,
    /// Score the recording's executor in place of miru.
    of_executor: bool,
    /// Give miru the truth as every view's masks.
    truth_masks: bool,
    /// Give miru each look's frames a second time.
    again: bool,
    /// What every view's stated noise is multiplied by; none for the noise as stated.
    noise: Option<f32>,
    /// Say what share of each kind's pixels is an object's, a known shape's and nothing's.
    shares: bool,
    /// Say how far a matched object's middle lies from the truth's, per kind.
    offsets: bool,
    /// The set of the same scenes on other depth, to hold each matched object against.
    twin: Option<PathBuf>,
    /// How many threads miru's look may use; none for as many as the machine has.
    threads: Option<usize>,
    /// Where to write each view's labels as a PNG.
    images: Option<PathBuf>,
    /// Print one hash over every scene miru's looks gave.
    digest: bool,
    /// The runs to score, or none for all.
    only: Vec<String>,
}

impl Asked {
    /// What `args` ask for, the program's name left out: refused where they ask of the executor
    /// what only miru's look has: masks, a second look, the noise and the scenes' digest.
    fn of(mut args: impl Iterator<Item = String>) -> Result<Self, Error> {
        let usage = "usage: score <set> [--recorded | --masks truth] [--again] [--noise factor] \
             [--shares] [--offsets] [--twin set] [--threads n] [--images dir] [--digest] [run ...]";
        let set = PathBuf::from(args.next().ok_or(usage)?);
        let (mut of_executor, mut truth_masks, mut again) = (false, false, false);
        let (mut images, mut only, mut shares, mut threads) = (None, vec![], false, None);
        let (mut offsets, mut digest, mut noise, mut twin) = (false, false, None, None);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--recorded" => of_executor = true,
                "--masks" if args.next().as_deref() == Some("truth") => truth_masks = true,
                "--again" => again = true,
                "--noise" => {
                    let factor: f32 = args.next().ok_or("--noise takes a factor")?.parse()?;
                    if !(factor.is_finite() && factor > 0.0) {
                        return Err(format!("--noise takes a factor over 0, not {factor}").into());
                    }
                    noise = Some(factor);
                }
                "--shares" => shares = true,
                "--offsets" => offsets = true,
                "--twin" => twin = Some(PathBuf::from(args.next().ok_or("--twin takes a set")?)),
                "--threads" => {
                    threads = Some(args.next().ok_or("--threads takes a number")?.parse()?);
                }
                "--images" => {
                    images = Some(PathBuf::from(args.next().ok_or("--images takes a folder")?))
                }
                "--digest" => digest = true,
                flag if flag.starts_with("--") => return Err(format!("{flag}? {usage}").into()),
                _ => only.push(arg),
            }
        }
        if of_executor && truth_masks {
            return Err(format!("masks are for miru's look, not the executor's: {usage}").into());
        }
        if of_executor && again {
            return Err(
                format!("only miru looks again, the executor's look is recorded: {usage}").into(),
            );
        }
        if of_executor && digest {
            return Err(
                format!("the digest is of miru's scenes, not the executor's: {usage}").into(),
            );
        }
        if of_executor && noise.is_some() {
            return Err(format!("the noise is stated to miru, not the executor: {usage}").into());
        }
        if of_executor && twin.is_some() {
            return Err(
                format!("only miru's shapes are held against a twin, the executor's are not recorded: {usage}")
                    .into(),
            );
        }
        Ok(Self {
            set,
            of_executor,
            truth_masks,
            again,
            noise,
            shares,
            offsets,
            twin,
            threads,
            images,
            digest,
            only,
        })
    }

    /// What each view's stated noise is multiplied by: the factor `--noise` gives, for the set
    /// scored, and none for its `twin`, whose views keep the noise their recording states, as the
    /// measure the shapes are held against.
    fn factor(&self, twin: bool) -> f32 {
        if twin { 1.0 } else { self.noise.unwrap_or(1.0) }
    }
}

/// A 64-bit FNV-1a hash, written out so that a digest is the same in every build and release.
struct Digest(u64);

impl Default for Digest {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Digest {
    fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }

    fn number(&mut self, number: usize) {
        self.bytes(&(number as u64).to_le_bytes());
    }

    fn floats(&mut self, floats: &[f32]) {
        for float in floats {
            self.bytes(&float.to_bits().to_le_bytes());
        }
    }

    /// Adds everything `scene` holds: each object's id, shape, whether it is solid, its colour,
    /// what it stands on and whether it was seen, the next id, and every view's labels.
    fn scene(&mut self, scene: &Scene) {
        self.number(scene.objects.len());
        for object in &scene.objects {
            self.number(object.id as usize);
            match object.shape {
                Shape::Cuboid {
                    center,
                    half_extents,
                    rotation,
                } => {
                    self.number(0);
                    self.floats(&center.to_array());
                    self.floats(&half_extents.to_array());
                    self.floats(&rotation.to_array());
                }
                Shape::Cylinder {
                    center,
                    rotation,
                    radius,
                    half_height,
                } => {
                    self.number(1);
                    self.floats(&center.to_array());
                    self.floats(&rotation.to_array());
                    self.floats(&[radius, half_height]);
                }
                Shape::OpenBox {
                    center,
                    half_extents,
                    rotation,
                    room_lower,
                    room_upper,
                } => {
                    self.number(2);
                    self.floats(&center.to_array());
                    self.floats(&half_extents.to_array());
                    self.floats(&rotation.to_array());
                    self.floats(&room_lower.to_array());
                    self.floats(&room_upper.to_array());
                }
                ref shape => panic!("a shape the scorer does not know: {shape:?}"),
            }
            self.number(usize::from(object.solid));
            match object.color {
                Some(color) => {
                    self.number(1);
                    self.bytes(&color);
                }
                None => self.number(0),
            }
            match object.on {
                Some(Support::Known(shape)) => {
                    self.number(1);
                    self.number(shape);
                }
                Some(Support::Object(below)) => {
                    self.number(2);
                    self.number(below);
                }
                None => self.number(0),
                Some(support) => panic!("a support the scorer does not know: {support:?}"),
            }
            self.number(usize::from(object.seen));
        }
        self.number(scene.next_id as usize);
        self.number(scene.labels.len());
        for labels in &scene.labels {
            self.number(labels.len());
            for label in labels {
                let (kind, index) = match *label {
                    Label::Nothing => (0, 0),
                    Label::Robot => (1, 0),
                    Label::Known(shape) => (2, shape),
                    Label::Object(object) => (3, object),
                    label => panic!("a label the scorer does not know: {label:?}"),
                };
                self.bytes(&[kind]);
                self.number(index);
            }
        }
    }
}

fn main() -> Result<(), Error> {
    let asked = Asked::of(std::env::args().skip(1))?;
    let (factor, twin_factor) = (
        asked.factor(/* twin */ false),
        asked.factor(/* twin */ true),
    );
    let Asked {
        set,
        of_executor,
        truth_masks,
        again,
        noise: _,
        shares,
        offsets,
        twin,
        threads,
        images,
        digest,
        only,
    } = asked;
    let device = common::device(threads)?;
    let mut scenes = Digest::default();
    let mut runs: Vec<PathBuf> = fs::read_dir(&set)?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter(|path| path.join("cell.json").exists())
        .filter(|path| only.is_empty() || only.iter().any(|run| path.ends_with(run)))
        .collect();
    runs.sort();
    let (mut together, mut by_camera) = (Tally::new(), BTreeMap::<String, Tally>::new());
    let mut labelled = BTreeMap::<String, Shares>::new();
    // Per kind, how far each matched one's middle lies from the truth's (meters), and how many
    // matched ones are solid.
    let mut off = BTreeMap::<String, (Vec<f32>, usize)>::new();
    // Of the objects given their frames again: how many there were, kept their ids, stayed unseen.
    let mut repeated = [0; 3];
    // Per kind, the matched objects held against their twins.
    let mut twins = BTreeMap::<String, Twins>::new();
    for run in &runs {
        // Only miru is told the known shapes.
        let known = if of_executor {
            vec![]
        } else {
            common::fixtures(run)?
        };
        // The looks that have their truth, by their number.
        let mut looks: Vec<(usize, PathBuf)> = fs::read_dir(run)?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .filter(|path| path.join("truth.json").exists())
            .filter_map(|path| Some((path.file_name()?.to_str()?.parse().ok()?, path)))
            .collect();
        looks.sort();
        let mut scene = Scene::default();
        // The twin's run: its fixtures, and the scene of its looks so far.
        let twin_run = twin
            .as_ref()
            .map(|twin| twin.join(run.strip_prefix(&set).unwrap_or(run)));
        let twin_known = match &twin_run {
            Some(twin_run) => common::fixtures(twin_run)?,
            None => vec![],
        };
        let mut twin_scene = Scene::default();
        for (_, folder) in &looks {
            let recorded = Recorded::load(folder)?;
            let name = folder.strip_prefix(&set)?.display().to_string();
            let named: Vec<Vec<Label>>;
            let (labels, measured) = if of_executor {
                println!("{name}: {} objects", recorded.measured);
                named = recorded.cameras.iter().map(executor_labels).collect();
                (&named, recorded.measured)
            } else {
                let start = Instant::now();
                scene = looked(&recorded, &known, truth_masks, factor, &device, &scene)?;
                let took = start.elapsed().as_secs_f64() * 1e3;
                println!("{name}: {} objects in {took:.1} ms", scene.objects.len());
                scenes.scene(&scene);
                if again {
                    let start = Instant::now();
                    let second = looked(&recorded, &known, truth_masks, factor, &device, &scene)?;
                    let took = start.elapsed().as_secs_f64() * 1e3;
                    scenes.scene(&second);
                    let counts = [
                        scene.objects.len(),
                        kept(&scene, &second),
                        second.objects.iter().filter(|object| !object.seen).count(),
                    ];
                    let [all, kept, unseen] = counts;
                    println!(
                        "  again: {kept} of {all} keep their ids, {unseen} unseen, {took:.1} ms"
                    );
                    repeated = std::array::from_fn(|k| repeated[k] + counts[k]);
                }
                (&scene.labels, scene.objects.len())
            };
            let truths = recorded.truth.len();
            let mut sum = Counts::none(truths, measured);
            for (camera, labels) in recorded.cameras.iter().zip(labels) {
                let counts = Counts::of(camera, labels, truths, measured);
                println!("  {}", camera.name);
                let tally = by_camera.entry(camera.name.clone()).or_default();
                report(&counts.verdicts(), &recorded.truth, tally);
                sum.add(&counts);
                if shares {
                    let shares = labelled.entry(camera.name.clone()).or_default();
                    share(camera, labels, &recorded.truth, shares);
                }
                if let Some(images) = &images {
                    let file =
                        images.join(format!("{}-{}.png", name.replace('/', "-"), camera.name));
                    fs::create_dir_all(images)?;
                    picture(&file, camera.width, labels)?;
                }
            }
            println!("  cameras together");
            let verdicts = sum.verdicts();
            report(&verdicts, &recorded.truth, &mut together);
            let centers: Vec<Vec2> = if of_executor {
                recorded.centers.clone()
            } else {
                scene
                    .objects
                    .iter()
                    .map(|object| middle(&object.shape))
                    .collect()
            };
            for (t, scored) in verdicts.iter().enumerate() {
                // One matched lies in the object it is matched with: that one holds most of it.
                if let Some(Scored {
                    verdict: Verdict::Matched,
                    most: Some(Most { object, .. }),
                    ..
                }) = scored
                {
                    let kind = kind(&recorded.truth[t]).to_owned();
                    let from = centers[*object].distance(recorded.origins[t]);
                    let solid = !of_executor && scene.objects[*object].solid;
                    let (offsets, solids) = off.entry(kind).or_default();
                    offsets.push(from);
                    *solids += usize::from(solid);
                }
            }
            if let Some(twin) = &twin {
                let other = Recorded::load(&twin.join(&name))?;
                if other.truth != recorded.truth {
                    return Err(format!("{name}: the twin shows other truth objects").into());
                }
                twin_scene = looked(
                    &other,
                    &twin_known,
                    truth_masks,
                    twin_factor,
                    &device,
                    &twin_scene,
                )?;
                let mut theirs = Counts::none(other.truth.len(), twin_scene.objects.len());
                for (camera, labels) in other.cameras.iter().zip(&twin_scene.labels) {
                    let counts =
                        Counts::of(camera, labels, other.truth.len(), twin_scene.objects.len());
                    theirs.add(&counts);
                }
                let (ours, theirs) = (matched(&verdicts), matched(&theirs.verdicts()));
                for (t, pair) in ours.iter().zip(&theirs).enumerate() {
                    let (Some(object), Some(twin)) = pair else {
                        continue;
                    };
                    let (object, twin) = (&scene.objects[*object], &twin_scene.objects[*twin]);
                    twins
                        .entry(kind(&recorded.truth[t]).to_owned())
                        .or_default()
                        .add(object, twin);
                }
            }
        }
    }
    if together.is_empty() {
        return Err(format!("no look with its truth scored in {}", set.display()).into());
    }
    table("cameras together", &together);
    if offsets {
        println!(
            "\n{:32} {:>7} {:>7} {:>9}",
            "offset, cameras together", "matched", "solid", "median"
        );
        for (kind, (off, solid)) in &off {
            let median = 100.0 * median(off);
            let solid = if of_executor {
                "-".to_owned()
            } else {
                solid.to_string()
            };
            println!("{kind:32} {:>7} {solid:>7} {median:>6.2} cm", off.len());
        }
    }
    if let Some(twin) = &twin {
        println!(
            "\n{:32} {:>7} {:>21} {:>21} {:>21} {:>9} {:>9}",
            format!("twin {}", twin.display()),
            "matched",
            "half width",
            "longest half side",
            "half height",
            "round",
            "solid"
        );
        println!(
            "{:32} {:>7} {:>21} {:>21} {:>21} {:>9} {:>9}",
            "",
            "",
            "median, most (mm)",
            "median, most (mm)",
            "median, most (mm)",
            "as twin",
            "as twin"
        );
        for (kind, twins) in &twins {
            println!("{kind:32} {}", twins.line());
        }
    }
    for (camera, tally) in &by_camera {
        table(camera, tally);
    }
    for (camera, shares) in &labelled {
        println!(
            "\n{camera:32} {:>9} {:>7} {:>7} {:>7}",
            "pixels", "object", "known", "nothing"
        );
        for (kind, counts) in shares {
            let pixels: usize = counts.iter().sum();
            let [object, known, nothing] = counts.map(|n| 100.0 * n as f64 / pixels as f64);
            println!("{kind:32} {pixels:>9} {object:>6.1}% {known:>6.1}% {nothing:>6.1}%");
        }
    }
    if again {
        let [all, kept, unseen] = repeated;
        println!(
            "\nthe same frames again: {kept} of {all} objects keep their ids, {unseen} stay unseen"
        );
    }
    if digest {
        println!("\ndigest of the scenes: {:016x}", scenes.0);
    }
    Ok(())
}

/// How many objects of `first` keep their ids in `second`, the scene of the same frames looked at
/// again after it. The same frames show the same objects in the same order, so each is held
/// against the one in its place: an id that another object of `first` had is not kept.
fn kept(first: &Scene, second: &Scene) -> usize {
    let pairs = first.objects.iter().zip(&second.objects);
    pairs.filter(|(was, is)| is.seen && is.id == was.id).count()
}

/// The scene miru makes of `recorded`'s views after `previous`, with `known` as its fixtures and,
/// with `truth_masks`, the truth as every view's masks, each view's stated noise multiplied by
/// `noise`, on `device`'s threads.
fn looked(
    recorded: &Recorded,
    known: &[Obstacle],
    truth_masks: bool,
    noise: f32,
    device: &Device,
    previous: &Scene,
) -> Result<Scene, Error> {
    // The truth as a mask model would give it: 0 where a pixel shows no object.
    let number = |truth: &Option<usize>| truth.map_or(0, |object| object as u32 + 1);
    let masks: Vec<Vec<u32>> = (recorded.cameras.iter())
        .filter(|_| truth_masks)
        .map(|camera| camera.truth.iter().map(number).collect())
        .collect();
    let views: Vec<View<'_>> = (recorded.cameras.iter().enumerate())
        .map(|(index, camera)| View {
            image: DepthImage {
                depth: &camera.depth,
                width: camera.width,
                intrinsics: camera.intrinsics,
                camera: camera.pose,
            },
            color: &camera.color,
            noise: Noise {
                constant: noise * camera.noise.constant,
                linear: noise * camera.noise.linear,
                quadratic: noise * camera.noise.quadratic,
            },
            robot: &camera.robot,
            masks: masks.get(index).map_or(&[], Vec::as_slice),
        })
        .collect();
    let look = Look {
        views: &views,
        known,
    };
    // Readings count however far they lie: a recording does not say how far its robot reaches,
    // and its fixtures take in the floor and the walls.
    Ok(miru::look(
        device,
        &look,
        previous,
        &LookOptions::default(),
    )?)
}

/// Per truth object of `verdicts`, the measured object it is matched with, if it is.
fn matched(verdicts: &[Option<Scored>]) -> Vec<Option<usize>> {
    let matched = verdicts.iter().map(|scored| match scored {
        Some(Scored {
            verdict: Verdict::Matched,
            most: Some(Most { object, .. }),
            ..
        }) => Some(*object),
        _ => None,
    });
    matched.collect()
}

/// What a shape takes up, as half sizes (meters): from above, the mean of its half widths over
/// the directions there and its longest half side, and its half height.
fn halves(shape: &Shape) -> [f32; 3] {
    match *shape {
        Shape::Cuboid { half_extents, .. } | Shape::OpenBox { half_extents, .. } => [
            std::f32::consts::FRAC_2_PI * (half_extents.x + half_extents.y),
            half_extents.x.max(half_extents.y),
            half_extents.z,
        ],
        Shape::Cylinder {
            radius,
            half_height,
            ..
        } => [radius, radius, half_height],
        _ => panic!("a shape the scorer does not know: {shape:?}"),
    }
}

/// Of one kind, the matched objects held against their twins.
#[derive(Default)]
struct Twins {
    /// Per object, how much each of its [`halves`] exceeds its twin's (meters).
    excess: Vec<[f32; 3]>,
    /// How many twins are round, and of them how many objects are.
    round: [usize; 2],
    /// How many twins are solid, and of them how many objects are.
    solid: [usize; 2],
}

impl Twins {
    fn add(&mut self, object: &miru::Object, twin: &miru::Object) {
        let (ours, theirs) = (halves(&object.shape), halves(&twin.shape));
        self.excess
            .push(std::array::from_fn(|k| ours[k] - theirs[k]));
        let round = |shape: &Shape| matches!(shape, Shape::Cylinder { .. });
        if round(&twin.shape) {
            self.round[0] += 1;
            self.round[1] += usize::from(round(&object.shape));
        }
        if twin.solid {
            self.solid[0] += 1;
            self.solid[1] += usize::from(object.solid);
        }
    }

    /// How many, each excess's median and largest in millimeters, and the counts as twin.
    fn line(&self) -> String {
        let excesses = (0..3).map(|k| {
            let excess: Vec<f32> = self.excess.iter().map(|e| 1e3 * e[k]).collect();
            let most = excess.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            format!("{:>+10.1} {:>+10.1}", median(&excess), most)
        });
        let [twins_round, round] = self.round;
        let [twins_solid, solid] = self.solid;
        format!(
            "{:>7} {} {:>9} {:>9}",
            self.excess.len(),
            excesses.collect::<Vec<_>>().join(" "),
            format!("{round} of {twins_round}"),
            format!("{solid} of {twins_solid}")
        )
    }
}

/// Adds to `shares` what `labels` make of `camera`'s pixels that have a reading which is not
/// the robot's: per kind of the truth object each shows, of those `names` names, whether it is
/// an object's, a known shape's or nothing's.
fn share(camera: &Camera, labels: &[Label], names: &[String], shares: &mut Shares) {
    // Per truth object, and last for the pixels that show none.
    let mut counts = vec![[0; 3]; names.len() + 1];
    for (i, label) in labels.iter().enumerate() {
        if camera.depth[i] > 0.0 && !camera.robot[i] {
            let made = match label {
                Label::Object(_) => 0,
                Label::Known(_) => 1,
                _ => 2,
            };
            counts[camera.truth[i].unwrap_or(names.len())][made] += 1;
        }
    }
    for (truth, counts) in counts.iter().enumerate() {
        let kind = names.get(truth).map_or(NO_OBJECT, |name| kind(name));
        let sum = shares.entry(kind.to_owned()).or_default();
        *sum = std::array::from_fn(|k| sum[k] + counts[k]);
    }
}

/// What the executor that recorded `camera`'s frame made of each pixel, as miru's labels: one of
/// the objects it measured, the robot, or nothing.
fn executor_labels(camera: &Camera) -> Vec<Label> {
    let pixels = camera.measured.iter().zip(&camera.robot);
    pixels
        .map(|(object, &robot)| match *object {
            Some(object) => Label::Object(object),
            None if robot => Label::Robot,
            None => Label::Nothing,
        })
        .collect()
}

/// How many pixels with a reading show each truth object and are each measured object's: one
/// camera's, or those of a look's cameras added up.
struct Counts {
    /// Per truth object and measured object, the pixels that show the one and are the other's.
    shared: Vec<Vec<usize>>,
    /// Per truth object, the pixels that show it, in a measured object or in none.
    pixels: Vec<usize>,
    /// Per truth object, those of them under the robot's mask.
    masked: Vec<usize>,
}

/// How a truth object in view came out.
enum Verdict {
    Matched,
    /// With this truth object.
    Merged(usize),
    /// This many.
    Pieces(usize),
    Masked,
    Missed,
}

impl Verdict {
    /// Its place among `VERDICTS`.
    fn column(&self) -> usize {
        match self {
            Verdict::Matched => 0,
            Verdict::Merged(_) => 1,
            Verdict::Pieces(_) => 2,
            Verdict::Masked => 3,
            Verdict::Missed => 4,
        }
    }
}

/// A truth object in view, scored.
struct Scored {
    verdict: Verdict,
    /// The pixels with a reading that show it.
    pixels: usize,
    /// The measured object that holds most of it, if any holds a pixel of it.
    most: Option<Most>,
}

/// The measured object that holds most of a truth object.
struct Most {
    object: usize,
    /// The share of the truth object's pixels in it.
    held: f64,
    /// The share of its pixels on truth objects that are this one's.
    pure: f64,
}

impl Counts {
    /// No pixels of `truths` truth objects and `measured` measured ones.
    fn none(truths: usize, measured: usize) -> Self {
        Self {
            shared: vec![vec![0; measured]; truths],
            pixels: vec![0; truths],
            masked: vec![0; truths],
        }
    }

    /// The pixels of `camera`, each of which `labels` says the measured object of, if any.
    fn of(camera: &Camera, labels: &[Label], truths: usize, measured: usize) -> Self {
        let mut counts = Self::none(truths, measured);
        for (i, truth) in camera.truth.iter().enumerate() {
            let Some(truth) = *truth else { continue };
            if camera.depth[i] > 0.0 {
                counts.pixels[truth] += 1;
                counts.masked[truth] += usize::from(camera.robot[i]);
                if let Label::Object(object) = labels[i] {
                    counts.shared[truth][object] += 1;
                }
            }
        }
        counts
    }

    /// Adds the pixels of `other`, counted of the same objects.
    fn add(&mut self, other: &Self) {
        let shared = (self.shared.iter_mut().flatten()).zip(other.shared.iter().flatten());
        let pixels = self.pixels.iter_mut().zip(&other.pixels);
        let masked = self.masked.iter_mut().zip(&other.masked);
        for (to, from) in shared.chain(pixels).chain(masked) {
            *to += from;
        }
    }

    /// Per truth object how it came out, or None for one not in view. Where two would do, the
    /// first in order is taken, as the scorecard takes it.
    fn verdicts(&self) -> Vec<Option<Scored>> {
        let Self {
            shared,
            pixels,
            masked,
        } = self;
        let (truths, measured) = (0..pixels.len(), 0..shared.first().map_or(0, Vec::len));
        // Per measured object, its pixels on truth objects, and the truth object it is of.
        let shown: Vec<usize> = (measured.clone())
            .map(|o| shared.iter().map(|row| row[o]).sum())
            .collect();
        let of: Vec<Option<usize>> = (measured.clone())
            .map(|o| truths.clone().find(|&t| 2 * shared[t][o] > shown[o]))
            .collect();
        let lies_in = |t: usize, o: usize| pixels[t] >= IN_VIEW && 2 * shared[t][o] > pixels[t];
        let scored = truths.clone().map(|t| {
            if pixels[t] < IN_VIEW {
                return None;
            }
            let own: Vec<usize> = measured.clone().filter(|&o| of[o] == Some(t)).collect();
            let into = (measured.clone()).find(|&o| lies_in(t, o) && of[o] != Some(t));
            let taken = (own.iter()).find_map(|&o| {
                truths
                    .clone()
                    .find(|&other| other != t && lies_in(other, o))
            });
            let verdict = if 2 * masked[t] > pixels[t] {
                Verdict::Masked
            } else if let Some(into) = into {
                // With the one that measured object is of, or, where it is of none, the other
                // one most of it shows.
                let others = truths.clone().filter(|&other| other != t);
                let other = of[into]
                    .or_else(|| others.max_by_key(|&other| (shared[other][into], Reverse(other))))
                    .expect("a measured object that is of none shows two truth objects or more");
                Verdict::Merged(other)
            } else if let Some(other) = taken {
                Verdict::Merged(other)
            } else if own.len() > 1 {
                Verdict::Pieces(own.len())
            } else if own.first().is_some_and(|&o| lies_in(t, o)) {
                Verdict::Matched
            } else {
                Verdict::Missed
            };
            let most = (measured.clone())
                .filter(|&o| shared[t][o] > 0)
                .max_by_key(|&o| (shared[t][o], Reverse(o)));
            Some(Scored {
                verdict,
                pixels: pixels[t],
                most: most.map(|object| Most {
                    object,
                    held: shared[t][object] as f64 / pixels[t] as f64,
                    pure: shared[t][object] as f64 / shown[object] as f64,
                }),
            })
        });
        scored.collect()
    }
}

/// Prints each verdict of `scored`, which is per truth object of `names`, and adds it to `tally`
/// under the object's kind.
fn report(scored: &[Option<Scored>], names: &[String], tally: &mut Tally) {
    for (name, scored) in names.iter().zip(scored) {
        let Some(scored) = scored else { continue };
        tally.entry(kind(name).to_owned()).or_default()[scored.verdict.column()] += 1;
        let said = match scored.verdict {
            Verdict::Merged(with) => format!("merged with {}", names[with]),
            Verdict::Pieces(pieces) => format!("{pieces} pieces"),
            ref plain => VERDICTS[plain.column()].to_owned(),
        };
        let most = match &scored.most {
            Some(Most { object, held, pure }) => {
                format!("object {object} held {held:.3} pure {pure:.3}")
            }
            None => "in no object".to_owned(),
        };
        println!("    {name}: {said}; {} pixels, {most}", scored.pixels);
    }
}

/// A shape's middle seen from above.
fn middle(shape: &Shape) -> Vec2 {
    match *shape {
        Shape::Cuboid { center, .. }
        | Shape::Cylinder { center, .. }
        | Shape::OpenBox { center, .. } => center.truncate(),
        _ => panic!("a shape the scorer does not know: {shape:?}"),
    }
}

/// The median of `values`, which hold one at least: of an even number, the mean of the two in
/// the middle, as Python's `statistics.median` takes it.
fn median(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let half = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[half]
    } else {
        0.5 * (sorted[half - 1] + sorted[half])
    }
}

/// A truth object's kind: its name up to its last underscore, so without the number that tells
/// one of a kind from the next (`akita_black_bowl_2` is an `akita_black_bowl`).
fn kind(name: &str) -> &str {
    name.rsplit_once('_').map_or(name, |(kind, _)| kind)
}

/// Prints `tally` under `heading`: a line per kind, those most often in view first, and their
/// sum.
fn table(heading: &str, tally: &Tally) {
    let [matched, merged, pieces, masked, missed] = VERDICTS;
    println!("\n{heading:32} in view {matched:>7} {merged:>7} {pieces:>7} {masked:>7} {missed:>7}");
    let mut kinds: Vec<(&str, [usize; 5])> = (tally.iter())
        .map(|(kind, verdicts)| (kind.as_str(), *verdicts))
        .collect();
    kinds.sort_by_key(|(_, verdicts)| Reverse(verdicts.iter().sum::<usize>()));
    let mut all = [0; 5];
    for (_, verdicts) in &kinds {
        for (sum, n) in all.iter_mut().zip(verdicts) {
            *sum += n;
        }
    }
    kinds.push(("all", all));
    for (kind, [matched, merged, pieces, masked, missed]) in kinds {
        let in_view = matched + merged + pieces + masked + missed;
        println!(
            "{kind:32} {in_view:>7} {matched:>7} {merged:>7} {pieces:>7} {masked:>7} {missed:>7}"
        );
    }
}

/// Writes `labels` as a PNG: each object in a colour of its own, known shapes dark gray, the
/// robot light gray, nothing black.
fn picture(file: &Path, width: usize, labels: &[Label]) -> Result<(), Error> {
    let color = |label: &Label| match *label {
        Label::Object(object) => {
            // Hues a golden ratio's turn apart, so neighbors in number differ in colour.
            let hue = (object as f32 * 0.618_034).fract() * 6.0;
            let ramp = |shift: f32| {
                let x = (hue + shift).rem_euclid(6.0);
                (255.0 * (x.min(4.0 - x)).clamp(0.0, 1.0)) as u8
            };
            let light = if object % 2 == 0 { 0 } else { 90 };
            [ramp(4.0), ramp(2.0), ramp(0.0)].map(|c: u8| c.saturating_add(light).max(60))
        }
        Label::Known(_) => [50, 50, 50],
        Label::Robot => [200, 200, 200],
        _ => [0, 0, 0],
    };
    let pixels: Vec<u8> = labels.iter().flat_map(color).collect();
    let mut encoder = png::Encoder::new(
        BufWriter::new(fs::File::create(file)?),
        width as u32,
        (labels.len() / width) as u32,
    );
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use batchplan::{Intrinsics, Pose};
    use glam::{Quat, Vec2, Vec3};
    use miru::Noise;

    use super::*;

    /// A recorded look of one camera 1 m over a table, looking straight down at two slabs 10 cm
    /// tall on it, which states 2.5 mm of noise at 1 m.
    fn slabs() -> Recorded {
        let mut depth = vec![1.0; 64 * 48];
        for (row, column) in (16..32).flat_map(|row| (8..24).chain(40..56).map(move |c| (row, c))) {
            depth[row * 64 + column] = 0.9;
        }
        let pixels = depth.len();
        let camera = Camera {
            name: "camera".to_owned(),
            width: 64,
            intrinsics: Intrinsics {
                fx: 60.0,
                fy: 60.0,
                cx: 31.5,
                cy: 23.5,
            },
            pose: Pose::new(
                Vec3::new(0.5, 0.0, 1.0),
                Quat::from_rotation_x(std::f32::consts::PI),
            ),
            noise: Noise {
                constant: 0.0,
                linear: 0.0,
                quadratic: 0.0025,
            },
            depth,
            color: vec![],
            robot: vec![false; pixels],
            measured: vec![None; pixels],
            truth: vec![None; pixels],
        };
        Recorded {
            cameras: vec![camera],
            measured: 0,
            centers: vec![],
            truth: vec![],
            origins: vec![],
        }
    }

    /// A table whose top is the plane z = 0 under [`slabs`]' camera.
    fn table() -> Obstacle {
        Obstacle::cuboid(Vec3::new(0.5, 0.0, -0.5), Vec3::new(4.0, 4.0, 0.5))
    }

    /// The scene of [`slabs`] at the noise it states, the table known.
    fn two_slabs() -> Scene {
        let device = common::device(None).expect("a device");
        let scene = looked(&slabs(), &[table()], false, 1.0, &device, &Scene::default());
        scene.expect("a sound look")
    }

    #[test]
    fn an_object_that_has_the_id_of_another_has_not_kept_its_own() {
        let first = two_slabs();
        assert_eq!(first.objects.len(), 2, "{:?}", first.objects);
        assert_eq!(kept(&first, &first), 2);
        // Each takes the other's id: both ids are in the first scene, and neither was kept.
        let mut second = first.clone();
        second.objects[0].id = first.objects[1].id;
        second.objects[1].id = first.objects[0].id;
        assert_eq!(kept(&first, &second), 0);
        // One no view saw has not kept its id either: it is counted as unseen.
        let mut second = first.clone();
        second.objects[1].seen = false;
        assert_eq!(kept(&first, &second), 1);
    }

    #[test]
    fn a_share_is_of_the_readings_that_are_not_the_robots() {
        // Five pixels of one thing and a table: one without a reading and one the robot's count
        // for nothing; of the thing's other two one is an object's and one a known shape's, and
        // the table's is nothing's.
        let camera = Camera {
            name: "camera".to_owned(),
            width: 5,
            intrinsics: Intrinsics {
                fx: 1.0,
                fy: 1.0,
                cx: 0.0,
                cy: 0.0,
            },
            pose: Pose::new(Vec3::ZERO, Quat::IDENTITY),
            noise: Noise {
                constant: 0.0,
                linear: 0.0,
                quadratic: 0.0025,
            },
            depth: vec![0.0, 1.0, 1.0, 1.0, 1.0],
            color: vec![],
            robot: vec![false, true, false, false, false],
            measured: vec![None; 5],
            truth: vec![Some(0), Some(0), Some(0), Some(0), None],
        };
        let labels = [
            Label::Object(0),
            Label::Robot,
            Label::Object(0),
            Label::Known(0),
            Label::Nothing,
        ];
        let mut shares = Shares::new();
        share(&camera, &labels, &["mug_1".to_owned()], &mut shares);
        let expected = [("mug", [1, 1, 0]), (NO_OBJECT, [0, 0, 1])];
        assert_eq!(
            shares,
            expected
                .map(|(kind, counts)| (kind.to_owned(), counts))
                .into()
        );
    }

    #[test]
    fn a_shape_is_held_against_its_twin_by_its_half_sizes() {
        // A cylinder's mean half width from above is its radius, and a box's of half sides a and b
        // is 2 (a + b) / pi: a square of half side a takes up 4 a / pi, more than the circle in it.
        let cylinder = Shape::Cylinder {
            center: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            radius: 0.03,
            half_height: 0.05,
        };
        let cuboid = Shape::Cuboid {
            center: Vec3::ZERO,
            half_extents: Vec3::new(0.03, 0.03, 0.04),
            rotation: Quat::from_rotation_z(0.3),
        };
        assert_eq!(halves(&cylinder), [0.03, 0.03, 0.05]);
        let [width, longest, height] = halves(&cuboid);
        assert!(
            (width - 0.12 / std::f32::consts::PI).abs() < 1e-7 && longest == 0.03 && height == 0.04
        );
        // Held against the cylinder as its twin, the box is larger by the corners of the square.
        let object = |shape: Shape, solid: bool| {
            let mut scene = two_slabs();
            scene.objects[0].shape = shape;
            scene.objects[0].solid = solid;
            scene.objects[0].clone()
        };
        let mut twins = Twins::default();
        twins.add(&object(cuboid, false), &object(cylinder, true));
        twins.add(&object(cylinder, true), &object(cylinder, true));
        assert_eq!((twins.round, twins.solid), ([2, 1], [2, 1]));
        assert!((twins.excess[0][0] - (0.12 / std::f32::consts::PI - 0.03)).abs() < 1e-7);
        assert_eq!(twins.excess[1], [0.0; 3]);
    }

    #[test]
    fn a_median_is_the_middle_value_or_the_mean_of_the_two_middle_ones() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[7.0]), 7.0);
    }

    #[test]
    fn what_only_miru_does_is_refused_of_the_executor() {
        let asked = |args: &[&str]| Asked::of(args.iter().map(|arg| (*arg).to_owned()));
        for miru_only in [
            &["--masks", "truth"][..],
            &["--again"],
            &["--digest"],
            &["--noise", "0.25"],
            &["--twin", "exact"],
        ] {
            let alone: Vec<&str> = [&["set"], miru_only].concat();
            assert!(asked(&alone).is_ok(), "{alone:?}");
            let recorded: Vec<&str> = [&["set", "--recorded"], miru_only].concat();
            let refused = asked(&recorded).err().map(|error| error.to_string());
            assert!(
                refused
                    .as_deref()
                    .is_some_and(|said| said.contains("executor")),
                "{recorded:?}: {refused:?}"
            );
        }
        let all = ["set", "--masks", "truth", "--again", "--digest", "run"];
        let both = asked(&all).expect("miru's alone");
        assert!(both.truth_masks && both.again && both.digest && !both.of_executor);
        assert_eq!(both.only, ["run"]);
        assert!(!asked(&["set"]).expect("a set alone").digest);
        let twin = asked(&["set", "--twin", "exact", "run"]).expect("a twin and a run");
        assert_eq!(twin.twin, Some(PathBuf::from("exact")));
        assert_eq!(twin.only, ["run"]);
        assert!(asked(&["set", "--twin"]).is_err(), "a twin needs its set");
    }

    #[test]
    fn a_scene_that_differs_by_one_bit_has_another_digest() {
        let first = two_slabs();
        let digest = |scene: &Scene| {
            let mut digest = Digest::default();
            digest.scene(scene);
            digest.0
        };
        assert_eq!(digest(&first), digest(&first.clone()));
        let mut moved = first.clone();
        if let Shape::Cuboid { center, .. } | Shape::Cylinder { center, .. } =
            &mut moved.objects[0].shape
        {
            center.x = f32::from_bits(center.x.to_bits() + 1);
        }
        assert_ne!(digest(&first), digest(&moved));
        let mut relabelled = first.clone();
        relabelled.labels[0][0] = Label::Nothing;
        assert_ne!(relabelled.labels, first.labels);
        assert_ne!(digest(&first), digest(&relabelled));
        // Every other thing a scene holds is in the digest too: each changed alone gives
        // another. A box open at the top's fields each changed from those of `open`.
        let open = |center: Vec3, half: Vec3, rotation: Quat, lower: Vec3, upper: Vec2| {
            move |scene: &mut Scene| {
                scene.objects[0].shape = Shape::OpenBox {
                    center,
                    half_extents: half,
                    rotation,
                    room_lower: lower,
                    room_upper: upper,
                };
            }
        };
        let (center, half, rotation) = (Vec3::ZERO, Vec3::new(0.1, 0.2, 0.3), Quat::IDENTITY);
        let (lower, upper) = (Vec3::new(-0.05, -0.1, -0.2), Vec2::new(0.05, 0.1));
        let opened = [
            (
                "a box open at the top",
                open(center, half, rotation, lower, upper),
            ),
            (
                "an open box's middle",
                open(Vec3::X, half, rotation, lower, upper),
            ),
            (
                "an open box's half size",
                open(center, 2.0 * half, rotation, lower, upper),
            ),
            (
                "an open box's turn",
                open(center, half, Quat::from_rotation_z(1.0), lower, upper),
            ),
            (
                "its room's lower corner",
                open(center, half, rotation, 0.5 * lower, upper),
            ),
            (
                "its room's upper corner",
                open(center, half, rotation, lower, 0.5 * upper),
            ),
        ];
        let mut seen = vec![digest(&first)];
        for (what, change) in opened {
            let mut changed = first.clone();
            change(&mut changed);
            let digest = digest(&changed);
            assert!(!seen.contains(&digest), "{what} is not in the digest");
            seen.push(digest);
        }
        type Change = (&'static str, fn(&mut Scene));
        let changes: [Change; 9] = [
            ("an id", |scene| scene.objects[0].id += 1),
            ("a box's half size", |scene| {
                let half = Vec3::new(0.1, 0.2, 0.3);
                let (center, rotation) = (Vec3::ZERO, Quat::IDENTITY);
                scene.objects[0].shape = Shape::Cuboid {
                    center,
                    half_extents: half,
                    rotation,
                };
            }),
            ("a cylinder for a box", |scene| {
                scene.objects[0].shape = Shape::Cylinder {
                    center: Vec3::ZERO,
                    rotation: Quat::IDENTITY,
                    radius: 0.1,
                    half_height: 0.2,
                };
            }),
            ("solid", |scene| {
                scene.objects[0].solid = !scene.objects[0].solid
            }),
            ("a colour", |scene| {
                scene.objects[0].color = Some([1, 2, 3]);
            }),
            ("no colour", |scene| scene.objects[0].color = None),
            ("what it stands on", |scene| {
                scene.objects[0].on = Some(Support::Object(1));
            }),
            ("seen", |scene| {
                scene.objects[0].seen = !scene.objects[0].seen
            }),
            ("the next id", |scene| scene.next_id += 1),
        ];
        for (what, change) in changes {
            let mut changed = first.clone();
            change(&mut changed);
            if changed == first {
                continue;
            }
            let digest = digest(&changed);
            assert!(!seen.contains(&digest), "{what} is not in the digest");
            seen.push(digest);
        }
        assert!(
            seen.len() >= 14,
            "{} of the changes changed the scene",
            seen.len()
        );
    }

    #[test]
    fn a_noise_factor_multiplies_every_views_stated_noise() {
        // The slabs stand 10 cm over the table at 1 m: apart from it at the stated 2.5 mm, and
        // within three deviations of it at a hundred times that, where they read on the table.
        let (known, device) = ([table()], common::device(None).expect("a device"));
        let objects = |factor: f32| {
            let scene = looked(&slabs(), &known, false, factor, &device, &Scene::default());
            scene.expect("a sound look").objects.len()
        };
        assert_eq!((objects(1.0), objects(100.0)), (2, 0));
    }

    #[test]
    fn the_twin_keeps_the_noise_its_recording_states() {
        let asked = |args: &[&str]| Asked::of(args.iter().map(|arg| (*arg).to_owned()));
        let both = asked(&["set", "--noise", "0.25", "--twin", "exact"]).expect("both");
        assert_eq!((both.factor(false), both.factor(true)), (0.25, 1.0));
        let stated = asked(&["set", "--twin", "exact"]).expect("a twin");
        assert_eq!((stated.factor(false), stated.factor(true)), (1.0, 1.0));
    }

    #[test]
    fn a_noise_factor_is_a_number_over_0() {
        let asked = |args: &[&str]| Asked::of(args.iter().map(|arg| (*arg).to_owned()));
        let stated = asked(&["set", "run"]).expect("the noise as stated");
        assert_eq!(stated.noise, None);
        let quarter = asked(&["set", "--noise", "0.25", "run"]).expect("a factor");
        assert_eq!(
            (quarter.noise, &quarter.only[..]),
            (Some(0.25), &["run".to_owned()][..])
        );
        assert!(asked(&["set", "--noise"]).is_err(), "no factor");
        for refused in ["less", "0", "-1", "inf", "NaN"] {
            assert!(asked(&["set", "--noise", refused]).is_err(), "{refused}");
        }
    }
}

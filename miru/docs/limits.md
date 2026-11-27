# What miru does not do

- It runs no mask model. Depth alone does not join a thing's parts that show no connection, as a
  rack's slabs; a model's masks do ([masks](masks.md)).
- Things that touch can read as one object, even on exact depth: two boxes side by side, or a
  plate on a box ([the truth table](status.md#the-truth-table)).
- Shapes are boxes, upright cylinders and boxes open at the top. A thing whose shape only names
  the room it takes up, as a bowl, is not solid (`Object::solid`).
- `Memory::world` gives a thing that is not solid as its room, one solid box or cylinder, unless
  the step names it in `Step::into` or reaches its room at a configuration of `Step::at` or a hand
  pose of `Step::reach`; only then is it the map's part, as the cameras saw into it.
- On a stereo camera's depth most boxes come out larger than its noise reaches, and few are
  solid: through the D435 an 80 by 120 mm box measures 131 by 157 mm, where the allowance is 24 mm
  ([the truth table](status.md#the-truth-table)).
- The map is of 1 cm voxels as far as the robot reaches from its frame, within the corners
  `MemoryOptions` gives it (`lower_corner`, `upper_corner`). A reach past about 2 m, as a robot's
  whose base travels, is too large a map and is refused: give its memory the cell's workspace, or
  a corner a few centimeters under its floor. Readings outside the corners are in no look and no
  map, so no plan keeps clear of what only they show; the shapes a cell states are obstacles
  wherever they lie.
- What no view sees stays as the last look had it, until a view shows its place. What stood on a
  picked thing is in no memory until a look finds it.
- Every tolerance comes from the noise the caller states for each camera, but for two: the 10
  degrees of a crease between two things (LCCP's, `CREASE` in `src/segment.rs`) and how far a
  grip from above may lean off a control's axis (`ALONG` in `src/turn.rs`). Stated far below the
  camera's real noise, a mesh's facets become surfaces of their own.
- Grips are checked against the latest readings only ([grips](grips.md#what-the-evidence-covers)).

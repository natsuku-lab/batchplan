// VAMP and OMPL in the planner comparison (`scripts/compare.py --planner vamp|ompl`), from C++ with no
// Python in the loop, on the problems' capsule scenes (VAMP takes boxes and capsules only; every planner
// in this comparison gets the same capsules):
// - vamp: RRT-Connect, then VAMP's simplification;
// - ompl: OMPL's RRTConnect, then `simplifySolution`, with VAMP's check of one configuration as the state
//   validity checker and OMPL's own motion validator.
// One pass: every problem once with one seed, on `threads` threads pinned to `cores`, each thread taking
// the next problem. Appends each query's record and the pass's to the run directory, and the outputs of
// the first repetition. OMPL's RNG can only be seeded once per process, hence one pass per process.
//
//   c++ -std=c++17 -O3 -march=native -ffp-contract=fast -fno-math-errno -fno-signed-zeros \
//     -fno-trapping-math -fno-rounding-math -I<vamp>/src/impl -I<eigen3> -I<nigh>/src -I<pdqsort> \
//     -I<ompl include> scripts/compare_cpp.cc -L<ompl lib> -lompl -lpthread -o compare_cpp
//   compare_cpp <problems.txt> <run_dir> <vamp|ompl> <seed> <rep> <threads> <cores> <range> <max_iterations>
//       <time_limit> <segment_fraction>
//
// problems.txt (`compare.py` writes it): one problem per line: id, start (7), goal (7), the number of
// boxes then each box's centre, Euler XYZ angles and half extents, the number of capsules then each
// capsule's two ends and radius.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

#include <vamp/collision/factory.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/planning/simplify.hh>
#include <vamp/random/halton.hh>
#include <vamp/robots/panda.hh>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/util/Console.h>
#include <ompl/util/RandomNumbers.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

using Robot = vamp::robots::Panda;
static constexpr std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;
using RRTC = vamp::planning::RRTC<Robot, rake, Robot::resolution>;
using Config = std::array<float, Robot::dimension>;
using Clock = std::chrono::steady_clock;

struct Problem
{
    std::string id;
    Config start, goal;
    std::vector<std::array<float, 9>> cuboids;   // centre, Euler XYZ, half extents
    std::vector<std::array<float, 7>> capsules;  // end, end, radius
};

struct Answer
{
    bool solved = false;
    std::string reason;
    double setup = 0, search = 0, post = 0, total = 0;
    std::vector<Config> path;
};

struct Settings
{
    std::string planner;
    unsigned seed = 0;
    float range = 1.0F;
    std::size_t max_iterations = 0;  // 0: VAMP's default
    double time_limit = 5.0;
    double segment_fraction = 0.01;  // OMPL's motion-check spacing, as a fraction of the space's extent
};

static auto seconds(Clock::time_point a, Clock::time_point b) -> double
{
    return std::chrono::duration<double>(b - a).count();
}

// The scene as `vamp.Environment`'s add_cuboid and add_capsule build it.
static auto environment(const Problem &p) -> EnvironmentVector
{
    EnvironmentInput e;
    for (const auto &c : p.cuboids)
    {
        auto s = vamp::collision::factory::cuboid::array({c[0], c[1], c[2]}, {c[3], c[4], c[5]}, {c[6], c[7], c[8]});
        if (s.axis_3_z == 1.)
            e.z_aligned_cuboids.emplace_back(s);
        else
            e.cuboids.emplace_back(s);
    }
    for (const auto &c : p.capsules)
    {
        auto s = vamp::collision::factory::cylinder::endpoints::array({c[0], c[1], c[2]}, {c[3], c[4], c[5]}, c[6]);
        if (s.xv == 0. and s.yv == 0.)
            e.z_aligned_capsules.emplace_back(s);
        else
            e.capsules.emplace_back(s);
    }
    e.sort();
    return EnvironmentVector(e);
}

static auto to_config(const Robot::Configuration &q) -> Config
{
    const auto padded = q.to_array();  // rounded up to whole SIMD rows
    Config c;
    std::copy_n(padded.begin(), c.size(), c.begin());
    return c;
}

// RRT-Connect, then VAMP's simplification, on one Halton sequence whose bases are rotated `seed` times.
static auto plan_vamp(const Problem &p, const Settings &s) -> Answer
{
    Answer a;
    const auto t0 = Clock::now();
    auto env = environment(p);
    auto rng = std::make_shared<vamp::rng::Halton<Robot>>();
    for (unsigned k = 0; k < s.seed; ++k)
        rng->rotate_bases();
    vamp::planning::RRTCSettings settings;
    settings.range = s.range;
    if (s.max_iterations > 0)
    {
        settings.max_iterations = s.max_iterations;
        settings.max_samples = s.max_iterations;
    }
    const auto t1 = Clock::now();
    auto result = RRTC::solve(Robot::Configuration(p.start), Robot::Configuration(p.goal), env, settings, rng);
    const auto t2 = Clock::now();
    a.setup = seconds(t0, t1);
    a.search = seconds(t1, t2);
    if (result.path.size() == 0)
    {
        a.reason = "no path";
        a.total = seconds(t0, t2);
        return a;
    }
    vamp::planning::SimplifySettings simplify_settings;
    auto simplified =
        vamp::planning::simplify<Robot, rake, Robot::resolution>(result.path, env, simplify_settings, rng);
    const auto t3 = Clock::now();
    a.post = seconds(t2, t3);
    a.total = seconds(t0, t3);
    a.solved = true;
    for (const auto &q : simplified.path)
        a.path.push_back(to_config(q));
    return a;
}

// OMPL's RRTConnect within the time limit, then `simplifySolution` for up to a second.
static auto plan_ompl(const Problem &p, const Settings &s) -> Answer
{
    Answer a;
    const auto t0 = Clock::now();
    const auto env = environment(p);
    auto space = std::make_shared<ob::RealVectorStateSpace>(Robot::dimension);
    Config zeros{}, ones{};
    ones.fill(1.0f);
    auto lower = Robot::Configuration(zeros), upper = Robot::Configuration(ones);
    Robot::scale_configuration(lower);
    Robot::scale_configuration(upper);
    const auto low = lower.to_array(), high = upper.to_array();
    ob::RealVectorBounds bounds(Robot::dimension);
    for (auto j = 0U; j < Robot::dimension; ++j)
    {
        bounds.setLow(j, low[j]);
        bounds.setHigh(j, high[j]);
    }
    space->setBounds(bounds);
    og::SimpleSetup setup(space);
    setup.setStateValidityChecker(
        [&env](const ob::State *state)
        {
            const auto *values = state->as<ob::RealVectorStateSpace::StateType>()->values;
            Config c;
            for (auto j = 0U; j < Robot::dimension; ++j)
                c[j] = static_cast<float>(values[j]);
            const auto q = Robot::Configuration(c);
            return vamp::planning::validate_motion<Robot, rake, 1>(q, q, env);
        });
    setup.getSpaceInformation()->setStateValidityCheckingResolution(s.segment_fraction);
    setup.setPlanner(std::make_shared<og::RRTConnect>(setup.getSpaceInformation()));
    ob::ScopedState<> start(space), goal(space);
    for (auto j = 0U; j < Robot::dimension; ++j)
    {
        start[j] = p.start[j];
        goal[j] = p.goal[j];
    }
    setup.setStartAndGoalStates(start, goal);
    setup.setup();
    const auto t1 = Clock::now();
    const bool solved = bool(setup.solve(s.time_limit)) && setup.haveExactSolutionPath();
    const auto t2 = Clock::now();
    a.setup = seconds(t0, t1);
    a.search = seconds(t1, t2);
    if (!solved)
    {
        a.reason = "no path";
        a.total = seconds(t0, t2);
        return a;
    }
    setup.simplifySolution(1.0);
    const auto t3 = Clock::now();
    a.post = seconds(t2, t3);
    a.total = seconds(t0, t3);
    a.solved = true;
    const auto &path = setup.getSolutionPath();
    for (std::size_t w = 0; w < path.getStateCount(); ++w)
    {
        const auto *values = path.getState(w)->as<ob::RealVectorStateSpace::StateType>()->values;
        Config c;
        for (auto j = 0U; j < Robot::dimension; ++j)
            c[j] = static_cast<float>(values[j]);
        a.path.push_back(c);
    }
    return a;
}

static auto read_problems(const std::string &file) -> std::vector<Problem>
{
    std::ifstream in(file);
    std::vector<Problem> problems;
    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream row(line);
        Problem p;
        row >> p.id;
        for (auto &v : p.start)
            row >> v;
        for (auto &v : p.goal)
            row >> v;
        std::size_t n;
        row >> n;
        p.cuboids.resize(n);
        for (auto &c : p.cuboids)
            for (auto &v : c)
                row >> v;
        row >> n;
        p.capsules.resize(n);
        for (auto &c : p.capsules)
            for (auto &v : c)
                row >> v;
        if (!row)
            throw std::runtime_error("could not read problem " + p.id);
        problems.push_back(std::move(p));
    }
    return problems;
}

static auto parse_cores(const std::string &text) -> std::vector<int>
{
    std::vector<int> cores;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ','))
    {
        const auto dash = part.find('-');
        if (dash == std::string::npos)
            cores.push_back(std::stoi(part));
        else
            for (int c = std::stoi(part.substr(0, dash)); c <= std::stoi(part.substr(dash + 1)); ++c)
                cores.push_back(c);
    }
    return cores;
}

static void pin(int core)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

auto main(int argc, char **argv) -> int
{
    if (argc != 12)
    {
        std::cerr << "usage: compare_cpp <problems.txt> <run_dir> <vamp|ompl> <seed> <rep> <threads> <cores> "
                     "<range> <max_iterations> <time_limit> <segment_fraction>\n";
        return 2;
    }
    const std::string run_dir = argv[2];
    Settings s;
    s.planner = argv[3];
    s.seed = std::stoul(argv[4]);
    const int rep = std::stoi(argv[5]);
    const unsigned threads = std::stoul(argv[6]);
    const auto cores = parse_cores(argv[7]);
    s.range = std::stof(argv[8]);
    s.max_iterations = std::stoul(argv[9]);
    s.time_limit = std::stod(argv[10]);
    s.segment_fraction = std::stod(argv[11]);
    if (s.planner == "ompl")
        ompl::RNG::setSeed(1'000'003UL * s.seed + 1);
    ompl::msg::setLogLevel(ompl::msg::LOG_WARN);
    const auto plan = s.planner == "vamp" ? plan_vamp : plan_ompl;
    const auto problems = read_problems(argv[1]);

    std::vector<Answer> answers(problems.size());
    std::atomic<std::size_t> next{0};
    std::atomic<unsigned> ready{0};
    std::atomic<bool> go{false};
    Clock::time_point released;
    std::vector<std::thread> pool;
    const auto started = Clock::now();
    for (unsigned k = 0; k < threads; ++k)
        pool.emplace_back(
            [&, k]
            {
                if (!cores.empty())
                    pin(cores[k % cores.size()]);
                plan(problems[0], s);  // warm up, untimed
                ++ready;
                while (!go.load())
                    std::this_thread::yield();
                for (auto i = next++; i < problems.size(); i = next++)
                    answers[i] = plan(problems[i], s);
            });
    while (ready.load() < threads)
        std::this_thread::yield();
    released = Clock::now();
    go = true;
    for (auto &th : pool)
        th.join();
    const auto finished = Clock::now();

    std::ofstream records(run_dir + "/records.jsonl", std::ios::app);
    records << std::setprecision(9);
    for (std::size_t i = 0; i < problems.size(); ++i)
    {
        const auto &a = answers[i];
        records << "{\"problem\": \"" << problems[i].id << "\", \"seed\": " << s.seed << ", \"rep\": " << rep
                << ", \"status\": \"" << (a.solved ? "solved" : "failed") << "\", \"reason\": \"" << a.reason
                << "\", \"phases\": {\"setup\": " << a.setup << ", \"search\": " << a.search
                << ", \"post\": " << a.post << ", \"total\": " << a.total << "}, \"planner\": \"" << s.planner
                << "\"}\n";
    }
    std::ofstream passes(run_dir + "/passes.jsonl", std::ios::app);
    passes << std::setprecision(9) << "{\"rep\": " << rep << ", \"seed\": " << s.seed
           << ", \"problems\": " << problems.size() << ", \"wall\": " << seconds(released, finished)
           << ", \"setup_wall\": " << seconds(started, finished) << "}\n";
    if (rep == 1)
    {
        // The first repetition's paths, as `compare.py`'s write_outputs writes them.
        std::ofstream bin(run_dir + "/outputs_seed" + std::to_string(s.seed) + ".bin", std::ios::binary);
        std::ofstream index(run_dir + "/outputs_seed" + std::to_string(s.seed) + ".json");
        index << "{";
        std::size_t offset = 0;
        bool first = true;
        for (std::size_t i = 0; i < problems.size(); ++i)
        {
            if (!answers[i].solved)
                continue;
            const auto &path = answers[i].path;
            for (const auto &q : path)
                bin.write(reinterpret_cast<const char *>(q.data()), sizeof(float) * q.size());
            index << (first ? "" : ", ") << "\"" << problems[i].id << "|path\": [" << offset << ", " << path.size()
                  << ", " << Robot::dimension << "]";
            offset += path.size() * Robot::dimension;
            first = false;
        }
        index << "}";
    }
    return 0;
}

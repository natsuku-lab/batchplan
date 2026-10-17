// VAMP and OMPL from C++, with no Python in the loop, on the problems
// `scripts/compare_planners.py data/robometrics vamp_problems.txt vamp-export` writes, as the Python
// comparison plans them:
// - vamp: RRT-Connect, then VAMP's simplification.
// - ompl: OMPL's RRTConnect, then `simplifySolution`, with VAMP's check of one configuration as the
//   state validity checker and OMPL's own motion validator.
// Each problem is timed alone, including building its scene; each set is also planned on every core,
// one problem per thread at a time, for throughput. Writes the comparison's JSON for `benchmark --check`.
//
//   c++ -std=c++17 -O3 -mcpu=native -ffp-contract=fast -fno-math-errno -fno-signed-zeros \
//     -fno-trapping-math -fno-rounding-math -flto -I<vamp>/src/impl -I<eigen-3.4.0> -I<nigh>/src \
//     -I<pdqsort> -I<ompl>/include/ompl-2.0 scripts/cpp_planners.cc -L<ompl>/lib -lompl -o cpp_planners
//   ./cpp_planners vamp_problems.txt compare_vamp_cpp.json vamp [max_iterations [range]]
//   ./cpp_planners vamp_problems.txt compare_ompl_cpp.json ompl
// With max_iterations, RRT-Connect gives up sooner than VAMP's default 100,000 (batchplan's is 10,000).
// RRT-Connect extends by VAMP's Panda range, 1 rad (`ROBOT_RRT_RANGES` in its Python package, which its
// benchmark script uses), unless given another; `RRTCSettings`'s own default is 2.
// (VAMP 0.6.4's own release flags; on x86-64 use -march=native and its x86 fast-math flags.)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <vamp/collision/factory.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/planning/simplify.hh>
#include <vamp/random/halton.hh>
#include <vamp/robots/panda.hh>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/util/Console.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

using Robot = vamp::robots::Panda;
static constexpr std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;
using RRTC = vamp::planning::RRTC<Robot, rake, Robot::resolution>;
using Config = std::array<float, Robot::dimension>;

struct Problem
{
    Config start, goal;
    std::vector<std::array<float, 9>> cuboids;   // centre, Euler XYZ, half extents
    std::vector<std::array<float, 7>> capsules;  // end, end, radius
};

struct Attempt
{
    bool solved = false;
    double seconds = 0.0;
    std::vector<Config> path;
};

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
        e.sort();
    }
    for (const auto &c : p.capsules)
    {
        auto s = vamp::collision::factory::cylinder::endpoints::array({c[0], c[1], c[2]}, {c[3], c[4], c[5]}, c[6]);
        if (s.xv == 0. and s.yv == 0.)
            e.z_aligned_capsules.emplace_back(s);
        else
            e.capsules.emplace_back(s);
        e.sort();
    }
    return EnvironmentVector(e);
}

// One problem as `vamp.panda.rrtc` then `vamp.panda.simplify` plan it, with VAMP's Panda range and
// otherwise default settings, one Halton sequence for both.
static std::size_t max_iterations = 0;  // 0: VAMP's default
static float range = 1.0F;

static auto plan_vamp(const Problem &p) -> Attempt
{
    const auto t = std::chrono::steady_clock::now();
    auto env = environment(p);
    auto rng = std::make_shared<vamp::rng::Halton<Robot>>();
    vamp::planning::RRTCSettings settings;
    settings.range = range;
    if (max_iterations > 0)
    {
        settings.max_iterations = max_iterations;
        settings.max_samples = max_iterations;
    }
    auto result = RRTC::solve(Robot::Configuration(p.start), Robot::Configuration(p.goal), env, settings, rng);
    Attempt a;
    if (result.path.size() > 0)
    {
        vamp::planning::SimplifySettings simplify_settings;
        auto simplified =
            vamp::planning::simplify<Robot, rake, Robot::resolution>(result.path, env, simplify_settings, rng);
        a.solved = true;
        for (const auto &q : simplified.path)
        {
            const auto padded = q.to_array();  // rounded up to whole SIMD rows
            Config c;
            std::copy_n(padded.begin(), c.size(), c.begin());
            a.path.push_back(c);
        }
    }
    a.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    return a;
}

// One problem as the Python comparison's `plan_ompl` plans it: a 5 s limit, then 1 s of simplification.
static auto plan_ompl(const Problem &p) -> Attempt
{
    const auto t = std::chrono::steady_clock::now();
    const auto env = environment(p);
    auto space = std::make_shared<ob::RealVectorStateSpace>(Robot::dimension);
    // `vamp.panda.lower_bounds()` and `upper_bounds()`.
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
    // `vamp.panda.validate`.
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
    setup.setPlanner(std::make_shared<og::RRTConnect>(setup.getSpaceInformation()));
    ob::ScopedState<> start(space), goal(space);
    for (auto j = 0U; j < Robot::dimension; ++j)
    {
        start[j] = p.start[j];
        goal[j] = p.goal[j];
    }
    setup.setStartAndGoalStates(start, goal);
    Attempt a;
    a.solved = bool(setup.solve(5.0)) && setup.haveExactSolutionPath();
    if (a.solved)
    {
        setup.simplifySolution(1.0);
        const auto &path = setup.getSolutionPath();
        for (std::size_t w = 0; w < path.getStateCount(); ++w)
        {
            const auto *values = path.getState(w)->as<ob::RealVectorStateSpace::StateType>()->values;
            Config c;
            for (auto j = 0U; j < Robot::dimension; ++j)
                c[j] = static_cast<float>(values[j]);
            a.path.push_back(c);
        }
    }
    a.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    return a;
}

auto main(int argc, char **argv) -> int
{
    const std::string chosen = argc > 3 ? argv[3] : "";
    if (!(argc == 4 || (argc <= 6 && chosen == "vamp")) || (chosen != "vamp" && chosen != "ompl"))
    {
        std::cerr << "usage: cpp_planners problems.txt out.json vamp [max_iterations [range]] | ompl\n";
        return 2;
    }
    if (argc >= 5)
        max_iterations = std::stoul(argv[4]);
    if (argc == 6)
        range = std::stof(argv[5]);
    ompl::msg::setLogLevel(ompl::msg::LOG_WARN);
    const auto plan = chosen == "vamp" ? plan_vamp : plan_ompl;
    std::ifstream in(argv[1]);
    std::size_t set_count;
    in >> set_count;
    std::vector<std::pair<std::string, std::vector<Problem>>> sets(set_count);
    for (auto &[name, problems] : sets)
    {
        std::size_t count;
        in >> name >> count;
        problems.resize(count);
        for (auto &p : problems)
        {
            for (auto &v : p.start)
                in >> v;
            for (auto &v : p.goal)
                in >> v;
            std::size_t n;
            in >> n;
            p.cuboids.resize(n);
            for (auto &c : p.cuboids)
                for (auto &v : c)
                    in >> v;
            in >> n;
            p.capsules.resize(n);
            for (auto &c : p.capsules)
                for (auto &v : c)
                    in >> v;
        }
    }
    if (!in)
    {
        std::cerr << "could not read " << argv[1] << "\n";
        return 1;
    }

    const auto threads = std::max(1u, std::thread::hardware_concurrency());
    std::ofstream out(argv[2]);
    std::string planner = chosen + "-cpp" + (max_iterations > 0 ? "-" + std::to_string(max_iterations) : "");
    if (chosen == "vamp" && range != 1.0F)
        planner += "-range" + std::to_string(static_cast<int>(range));
    out << std::setprecision(9) << "{\"machine\": \"C++, " << threads << " threads\", \"planners\": {\"" << planner
        << "\": {";
    for (std::size_t s = 0; s < sets.size(); ++s)
    {
        const auto &[name, problems] = sets[s];
        // One at a time, for each problem's time and path.
        std::vector<Attempt> alone(problems.size());
        for (std::size_t i = 0; i < problems.size(); ++i)
            alone[i] = plan(problems[i]);
        // Every core, each thread taking the next problem, for throughput.
        std::vector<Attempt> together(problems.size());
        std::atomic<std::size_t> next{0};
        const auto t = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (unsigned k = 0; k < threads; ++k)
            pool.emplace_back(
                [&]
                {
                    for (auto i = next++; i < problems.size(); i = next++)
                        together[i] = plan(problems[i]);
                });
        for (auto &th : pool)
            th.join();
        const auto wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        std::size_t solved = 0, differ = 0;
        for (std::size_t i = 0; i < problems.size(); ++i)
        {
            solved += alone[i].solved;
            differ += alone[i].solved != together[i].solved || alone[i].path != together[i].path;
        }
        std::cerr << name << ": " << solved << "/" << problems.size() << " solved, " << wall << " s on " << threads
                  << " threads" << (differ ? ", " + std::to_string(differ) + " paths differ between runs" : "") << "\n";

        out << (s ? ", " : "") << "\"" << name << "\": {\"wall_seconds\": " << wall << ", \"problems\": [";
        for (std::size_t i = 0; i < problems.size(); ++i)
        {
            const auto &a = alone[i];
            out << (i ? ", " : "") << "{\"solved\": " << (a.solved ? "true" : "false") << ", \"seconds\": " << a.seconds
                << ", \"path\": ";
            if (!a.solved)
            {
                out << "null}";
                continue;
            }
            out << "[";
            for (std::size_t w = 0; w < a.path.size(); ++w)
            {
                out << (w ? ", " : "") << "[";
                for (std::size_t j = 0; j < a.path[w].size(); ++j)
                    out << (j ? ", " : "") << a.path[w][j];
                out << "]";
            }
            out << "]}";
        }
        out << "]}";
    }
    out << "}}}\n";
    return 0;
}

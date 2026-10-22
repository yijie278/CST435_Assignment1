#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
#endif

using namespace std;

using Vector = vector<double>;

struct Point {
    Vector raw;
    Vector standardized;
    int cluster = -1;
};

struct Result {
    vector<Point> points;
    vector<Vector> centers;
    int iterations = 0;
    bool converged = false;
    double wcss = 0.0;
    double elapsed_ms = 0.0;
};

struct Config {
    string input = "data\\Mall_Customers.csv";
    string features =
        "Annual Income (k$),Spending Score (1-100)";
    size_t points = 100000;
    bool synthetic = false;
    int k = 5;
    int repeats = 5;
    int max_iterations = 100;
    double tolerance = 1e-4;
    unsigned seed = 2026;
    vector<int> threads{1, 2, 4, 8};
    string results = "results\\raw\\current\\results.csv";
    string clusters = "results\\clusters\\current\\clustered_customers.csv";
    string centroids = "results\\centroids\\current\\centroids.csv";
};

struct Measurement {
    string implementation;
    size_t dataset_size;
    int k;
    int threads;
    int repeat;
    int iterations;
    double elapsed_ms;
    double wcss;
    bool converged;
};

// ---------------------------------------------------------------------------
// General helpers
// ---------------------------------------------------------------------------

static string trim(string value) {
    const auto first = value.find_first_not_of(" \t\r\n\"'");
    if (first == string::npos) {
        return "";
    }

    const auto last = value.find_last_not_of(" \t\r\n\"'");
    return value.substr(first, last - first + 1);
}

static vector<string> split_names(const string& value) {
    vector<string> names;
    stringstream stream(value);
    string name;

    while (getline(stream, name, ',')) {
        names.push_back(trim(name));
    }

    return names;
}

static vector<string> parse_csv_fields(const string& line) {
    vector<string> fields;
    string current;
    bool quoted = false;

    for (size_t i = 0; i < line.size(); ++i) {
        const char character = line[i];

        if (character == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                current += '"';
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            fields.push_back(trim(current));
            current.clear();
        } else {
            current += character;
        }
    }

    fields.push_back(trim(current));
    return fields;
}

static double squared_distance(const Vector& first, const Vector& second) {
    double result = 0.0;

    for (size_t dimension = 0; dimension < first.size(); ++dimension) {
        const double difference = first[dimension] - second[dimension];
        result += difference * difference;
    }

    return result;
}

static int nearest_center(const Vector& point,
                          const vector<Vector>& centers) {
    int best_center = 0;
    double best_distance = squared_distance(point, centers[0]);

    for (size_t center = 1; center < centers.size(); ++center) {
        const double distance = squared_distance(point, centers[center]);
        if (distance < best_distance) {
            best_distance = distance;
            best_center = static_cast<int>(center);
        }
    }

    return best_center;
}

// ---------------------------------------------------------------------------
// Data loading and preprocessing
// ---------------------------------------------------------------------------

static vector<Point> generate_synthetic_data(size_t count,
                                                   unsigned seed,
                                                   size_t dimensions) {
    if (count == 0) {
        throw runtime_error("--points must be positive");
    }
    if (dimensions == 0) {
        throw runtime_error(
            "At least one synthetic feature is required.");
    }

    mt19937 generator(seed);
    normal_distribution<double> noise(0.0, 1.0);
    const double center_x[5] = {20, 25, 40, 60, 80};
    const double center_y[5] = {20, 80, 50, 50, 80};

    vector<Point> points;
    points.reserve(count);

    for (size_t index = 0; index < count; ++index) {
        const int cluster = static_cast<int>(index % 5);
        Vector raw(dimensions);

        for (size_t dimension = 0; dimension < dimensions; ++dimension) {
            const double center =
                dimension == 0
                    ? center_x[cluster]
                    : dimension == 1
                          ? center_y[cluster]
                          : (static_cast<double>(cluster) - 2.0) * 12.0;
            const double spread =
                dimension == 0 ? 7.0 : dimension == 1 ? 10.0 : 5.0;
            raw[dimension] = center + noise(generator) * spread;
        }

        points.push_back({raw, {}, -1});
    }

    shuffle(points.begin(), points.end(), generator);
    return points;
}

static vector<Point> read_customer_data(
    const string& path,
    const vector<string>& requested_features) {
    ifstream input(path);
    if (!input) {
        throw runtime_error(
            "Cannot open CSV: " + path +
            ". Put Mall_Customers.csv in data or pass --input path.");
    }

    string line;
    if (!getline(input, line)) {
        throw runtime_error("CSV is empty.");
    }

    const auto header = parse_csv_fields(line);
    vector<size_t> feature_indices;

    for (const auto& feature : requested_features) {
        const auto position =
            find(header.begin(), header.end(), feature);
        if (position == header.end()) {
            string message =
                "Column not found: '" + feature + "'. Available columns: ";
            for (size_t index = 0; index < header.size(); ++index) {
                message += (index == 0 ? "" : ", ") + header[index];
            }
            throw runtime_error(message);
        }
        feature_indices.push_back(
            static_cast<size_t>(position - header.begin()));
    }

    vector<Point> points;
    size_t line_number = 1;

    while (getline(input, line)) {
        ++line_number;
        if (trim(line).empty()) {
            continue;
        }

        const auto fields = parse_csv_fields(line);
        Vector raw;

        for (const size_t index : feature_indices) {
            if (index >= fields.size()) {
                throw runtime_error(
                    "Not enough columns at line " +
                    to_string(line_number));
            }

            try {
                size_t characters_used = 0;
                const double value =
                    stod(fields[index], &characters_used);
                if (characters_used != fields[index].size() ||
                    !isfinite(value)) {
                    throw runtime_error("invalid numeric value");
                }
                raw.push_back(value);
            } catch (...) {
                throw runtime_error(
                    "Selected feature must be numeric; invalid value at line " +
                    to_string(line_number) +
                    ". Exclude Gender and CustomerID.");
            }
        }

        points.push_back({raw, {}, -1});
    }

    if (points.empty()) {
        throw runtime_error("No data rows in CSV.");
    }

    return points;
}

static void standardize_points(vector<Point>& points) {
    const size_t dimensions = points[0].raw.size();
    Vector means(dimensions, 0.0);
    Vector standard_deviations(dimensions, 0.0);

    for (const auto& point : points) {
        for (size_t dimension = 0; dimension < dimensions; ++dimension) {
            means[dimension] += point.raw[dimension];
        }
    }

    for (double& mean : means) {
        mean /= static_cast<double>(points.size());
    }

    for (const auto& point : points) {
        for (size_t dimension = 0; dimension < dimensions; ++dimension) {
            const double difference = point.raw[dimension] - means[dimension];
            standard_deviations[dimension] += difference * difference;
        }
    }

    for (double& deviation : standard_deviations) {
        deviation = sqrt(deviation / static_cast<double>(points.size()));
    }

    for (double& deviation : standard_deviations) {
        if (deviation < 1e-12) {
            deviation = 1.0;
        }
    }

    for (auto& point : points) {
        point.standardized.resize(dimensions);
        for (size_t dimension = 0; dimension < dimensions; ++dimension) {
            point.standardized[dimension] =
                (point.raw[dimension] - means[dimension]) /
                standard_deviations[dimension];
        }
    }
}

// ---------------------------------------------------------------------------
// K-Means initialisation and result calculations
// ---------------------------------------------------------------------------

static vector<Vector> initialise_centers(const vector<Point>& points,
                                               int k,
                                               unsigned seed) {
    if (k < 1 || static_cast<size_t>(k) > points.size()) {
        throw runtime_error(
            "K must be between 1 and number of rows.");
    }

    mt19937 generator(seed);
    uniform_int_distribution<size_t> first_pick(
        0, points.size() - 1);
    vector<Vector> centers;
    vector<size_t> chosen_indices;
    chosen_indices.reserve(static_cast<size_t>(k));

    const size_t first_index = first_pick(generator);
    chosen_indices.push_back(first_index);
    centers.push_back(points[first_index].standardized);

    vector<double> nearest_distances(points.size(), 0.0);

    while (centers.size() < static_cast<size_t>(k)) {
        double total_distance = 0.0;

        for (size_t index = 0; index < points.size(); ++index) {
            double distance =
                squared_distance(points[index].standardized, centers[0]);
            for (size_t center = 1; center < centers.size(); ++center) {
                distance = min(
                    distance,
                    squared_distance(points[index].standardized,
                                     centers[center]));
            }
            nearest_distances[index] = distance;
            total_distance += distance;
        }

        size_t selected_index = points.size();
        if (total_distance > 0.0 && isfinite(total_distance)) {
            uniform_real_distribution<double> pick(
                0.0, total_distance);
            const double target = pick(generator);
            double accumulated = 0.0;

            for (size_t index = 0; index < points.size(); ++index) {
                accumulated += nearest_distances[index];
                if (accumulated >= target) {
                    selected_index = index;
                    break;
                }
            }
        }

        const bool invalid_selection = selected_index == points.size();
        const bool already_chosen =
            !invalid_selection &&
            find(chosen_indices.begin(), chosen_indices.end(),
                      selected_index) != chosen_indices.end();

        if (invalid_selection || already_chosen) {
            for (size_t index = 0; index < points.size(); ++index) {
                if (find(chosen_indices.begin(), chosen_indices.end(),
                              index) == chosen_indices.end()) {
                    selected_index = index;
                    break;
                }
            }
        }

        if (selected_index == points.size()) {
            break;
        }

        chosen_indices.push_back(selected_index);
        centers.push_back(points[selected_index].standardized);
    }

    return centers;
}

static void assign_points_to_centers(vector<Point>& points,
                                     const vector<Vector>& centers) {
    for (auto& point : points) {
        point.cluster = nearest_center(point.standardized, centers);
    }
}

static double calculate_wcss(const vector<Point>& points,
                             const vector<Vector>& centers) {
    double total = 0.0;

    for (const auto& point : points) {
        total += squared_distance(
            point.standardized, centers[static_cast<size_t>(point.cluster)]);
    }

    return total;
}

static vector<size_t> calculate_cluster_sizes(
    const vector<Point>& points,
    size_t cluster_count) {
    vector<size_t> counts(cluster_count, 0);

    for (const auto& point : points) {
        if (point.cluster >= 0 &&
            static_cast<size_t>(point.cluster) < cluster_count) {
            ++counts[static_cast<size_t>(point.cluster)];
        }
    }

    return counts;
}

// ---------------------------------------------------------------------------
// Sequential and OpenMP K-Means implementations
// ---------------------------------------------------------------------------

static Result run_sequential(const vector<Point>& source,
                             int k,
                             int max_iterations,
                             double tolerance,
                             unsigned seed) {
    Result result;
    result.points = source;
    result.centers = initialise_centers(source, k, seed);
    const size_t dimensions = source[0].standardized.size();
    const auto start = chrono::steady_clock::now();

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        vector<Vector> sums(static_cast<size_t>(k),
                                 Vector(dimensions, 0.0));
        vector<size_t> counts(static_cast<size_t>(k), 0);

        for (auto& point : result.points) {
            const int cluster = nearest_center(point.standardized,
                                                result.centers);
            point.cluster = cluster;

            for (size_t dimension = 0; dimension < dimensions;
                 ++dimension) {
                sums[static_cast<size_t>(cluster)][dimension] +=
                    point.standardized[dimension];
            }
            ++counts[static_cast<size_t>(cluster)];
        }

        double maximum_movement = 0.0;
        for (int cluster = 0; cluster < k; ++cluster) {
            if (counts[static_cast<size_t>(cluster)] == 0) {
                continue;
            }

            Vector next_center(dimensions);
            for (size_t dimension = 0; dimension < dimensions;
                 ++dimension) {
                next_center[dimension] =
                    sums[static_cast<size_t>(cluster)][dimension] /
                    static_cast<double>(
                        counts[static_cast<size_t>(cluster)]);
            }

            maximum_movement = max(
                maximum_movement,
                squared_distance(
                    next_center,
                    result.centers[static_cast<size_t>(cluster)]));
            result.centers[static_cast<size_t>(cluster)] =
                move(next_center);
        }

        result.iterations = iteration + 1;
        result.converged = maximum_movement <= tolerance * tolerance;
        if (result.converged) {
            break;
        }
    }

    const auto end = chrono::steady_clock::now();
    result.elapsed_ms =
        chrono::duration<double, milli>(end - start).count();
    assign_points_to_centers(result.points, result.centers);
    result.wcss = calculate_wcss(result.points, result.centers);
    return result;
}

static Result run_openmp(const vector<Point>& source,
                         int k,
                         int max_iterations,
                         double tolerance,
                         unsigned seed,
                         int thread_count) {
    Result result;
    result.points = source;
    result.centers = initialise_centers(source, k, seed);

    const size_t point_count = source.size();
    const size_t dimensions = source[0].standardized.size();
    vector<Vector> global_sums(static_cast<size_t>(k),
                                    Vector(dimensions, 0.0));
    vector<long long> global_counts(static_cast<size_t>(k), 0);
    double maximum_movement = 0.0;
    const auto start = chrono::steady_clock::now();

    // Keep one OpenMP team alive for all iterations so worker creation and
    // destruction do not occur repeatedly inside the K-Means loop.
#pragma omp parallel num_threads(thread_count) \
    shared(global_sums, global_counts, maximum_movement, result)
    {
        vector<Vector> local_sums(static_cast<size_t>(k),
                                       Vector(dimensions, 0.0));
        vector<long long> local_counts(static_cast<size_t>(k), 0);

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            for (auto& row : local_sums) {
                fill(row.begin(), row.end(), 0.0);
            }
            fill(local_counts.begin(), local_counts.end(), 0);

#pragma omp single
            {
                for (auto& row : global_sums) {
                    fill(row.begin(), row.end(), 0.0);
                }
                fill(global_counts.begin(), global_counts.end(), 0);
                maximum_movement = 0.0;
            }

            // The implicit barrier at the end of single publishes the reset
            // accumulators before workers begin assigning points.
#pragma omp for schedule(static)
            for (long long index = 0;
                 index < static_cast<long long>(point_count); ++index) {
                auto& point = result.points[static_cast<size_t>(index)];
                const int cluster =
                    nearest_center(point.standardized, result.centers);
                point.cluster = cluster;

                for (size_t dimension = 0; dimension < dimensions;
                     ++dimension) {
                    local_sums[static_cast<size_t>(cluster)][dimension] +=
                        point.standardized[dimension];
                }
                ++local_counts[static_cast<size_t>(cluster)];
            }

            // Merge thread-local sums and counts without concurrent writes to
            // the shared accumulators.
#pragma omp critical(kmeans_merge)
            {
                for (int cluster = 0; cluster < k; ++cluster) {
                    global_counts[static_cast<size_t>(cluster)] +=
                        local_counts[static_cast<size_t>(cluster)];
                    for (size_t dimension = 0; dimension < dimensions;
                         ++dimension) {
                        global_sums[static_cast<size_t>(cluster)]
                                  [dimension] +=
                            local_sums[static_cast<size_t>(cluster)]
                                      [dimension];
                    }
                }
            }

#pragma omp barrier
#pragma omp single
            {
                for (int cluster = 0; cluster < k; ++cluster) {
                    if (global_counts[static_cast<size_t>(cluster)] == 0) {
                        continue;
                    }

                    Vector next_center(dimensions);
                    for (size_t dimension = 0; dimension < dimensions;
                         ++dimension) {
                        next_center[dimension] =
                            global_sums[static_cast<size_t>(cluster)]
                                      [dimension] /
                            static_cast<double>(
                                global_counts[static_cast<size_t>(
                                    cluster)]);
                    }

                    maximum_movement = max(
                        maximum_movement,
                        squared_distance(
                            next_center,
                            result.centers[static_cast<size_t>(cluster)]));
                    result.centers[static_cast<size_t>(cluster)] =
                        move(next_center);
                }

                result.iterations = iteration + 1;
                result.converged = maximum_movement <= tolerance * tolerance;
            }

            // The implicit barrier at the end of single publishes updated
            // centers and convergence state before the next iteration.
            if (result.converged) {
                break;
            }
        }
    }

    const auto end = chrono::steady_clock::now();
    result.elapsed_ms =
        chrono::duration<double, milli>(end - start).count();
    assign_points_to_centers(result.points, result.centers);
    result.wcss = calculate_wcss(result.points, result.centers);
    return result;
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

static string csv_quote(const string& value) {
    string quoted = "\"";
    for (const char character : value) {
        if (character == '"') {
            quoted += '"';
        }
        quoted += character;
    }
    quoted += '"';
    return quoted;
}

static void save_measurements(const string& path,
                              const vector<Measurement>& measurements) {
    double sequential_average = 0.0;
    int sequential_count = 0;
    for (const auto& measurement : measurements) {
        if (measurement.implementation == "sequential") {
            sequential_average += measurement.elapsed_ms;
            ++sequential_count;
        }
    }
    if (sequential_count > 0) {
        sequential_average /= sequential_count;
    }

    ofstream output(path);
    if (!output) {
        throw runtime_error("Cannot write " + path);
    }

    output << "implementation,dataset_size,k,threads,repeat,time_ms,"
              "sequential_average_ms,speedup,efficiency_percent,iterations,"
              "wcss,converged\n"
           << setprecision(12);

    for (const auto& measurement : measurements) {
        const double speedup =
            measurement.implementation == "sequential"
                ? 1.0
                : measurement.elapsed_ms > 0.0
                      ? sequential_average / measurement.elapsed_ms
                      : 0.0;
        const double efficiency =
            measurement.implementation == "sequential"
                ? 100.0
                : measurement.threads > 0
                      ? 100.0 * speedup / measurement.threads
                      : 0.0;

        output << measurement.implementation << ','
               << measurement.dataset_size << ',' << measurement.k << ','
               << measurement.threads << ',' << measurement.repeat << ','
               << measurement.elapsed_ms << ',' << sequential_average << ','
               << speedup << ',' << efficiency << ','
               << measurement.iterations << ',' << measurement.wcss << ','
               << (measurement.converged ? "true" : "false") << '\n';
    }
}

static void save_cluster_assignments(
    const string& path,
    const Result& result,
    const vector<string>& feature_names) {
    ofstream output(path);
    if (!output) {
        throw runtime_error("Cannot write " + path);
    }

    for (size_t index = 0; index < feature_names.size(); ++index) {
        output << csv_quote(feature_names[index]) << ',';
    }
    output << "cluster\n" << setprecision(12);

    for (const auto& point : result.points) {
        for (const double value : point.raw) {
            output << value << ',';
        }
        output << point.cluster << '\n';
    }
}

static void save_centers(const string& path,
                         const Result& result,
                         const vector<string>& feature_names) {
    ofstream output(path);
    if (!output) {
        throw runtime_error("Cannot write " + path);
    }

    output << "cluster";
    for (const auto& name : feature_names) {
        output << ',' << csv_quote("standardized_" + name);
    }
    output << '\n' << setprecision(12);

    for (size_t cluster = 0; cluster < result.centers.size(); ++cluster) {
        output << cluster;
        for (const double value : result.centers[cluster]) {
            output << ',' << value;
        }
        output << '\n';
    }
}

// ---------------------------------------------------------------------------
// Command-line interface
// ---------------------------------------------------------------------------

static vector<int> parse_threads(const string& value) {
    vector<int> thread_counts;
    stringstream stream(value);
    string token;

    while (getline(stream, token, ',')) {
        const int count = stoi(trim(token));
        if (count < 1) {
            throw runtime_error("Thread counts must be positive.");
        }
        if (find(thread_counts.begin(), thread_counts.end(), count) ==
            thread_counts.end()) {
            thread_counts.push_back(count);
        }
    }

    return thread_counts;
}

static void print_help() {
    cout
        << "CST 435 Parallel K-Means: customer segmentation\n\n"
        << "Options:\n"
        << "  --input FILE.csv       Dataset path (default data\\\\Mall_Customers.csv)\n"
        << "  --features A,B         Numeric column names, comma-separated\n"
        << "  --synthetic            Use generated data instead of CSV\n"
        << "  --points N             Synthetic points (default 100000)\n"
        << "  --k K                  Number of clusters (default 5)\n"
        << "  --threads 1,2,4,8      OpenMP thread counts\n"
        << "  --repeats N            Repetitions per configuration (default 5)\n"
        << "  --max-iterations N     Maximum iteration limit (default 100)\n"
        << "  --tolerance X          Centroid movement tolerance (default 0.0001)\n"
        << "  --seed N               Fixed seed (default 2026)\n"
        << "  --results FILE.csv     Benchmark output\n"
        << "  --clusters FILE.csv    Point/cluster output\n"
        << "  --centroids FILE.csv   Standardized centroid output\n"
        << "  --help                 Show help\n\n"
        << "Mall Customers example:\n"
        << "  ParallelKMeans.exe --input Mall_Customers.csv --features "
           "\"Annual Income (k$),Spending Score (1-100)\" --k 5 "
           "--threads 1,2,4 --repeats 5\n\n"
        << "Synthetic scalability example:\n"
        << "  ParallelKMeans.exe --synthetic --points 100000 --k 5 "
           "--threads 1,2,4,8 --repeats 5\n";
}

static Config parse_arguments(int argc, char** argv) {
    Config config;

    for (int index = 1; index < argc; ++index) {
        const string argument = argv[index];
        const auto next_value = [&]() {
            if (index + 1 >= argc) {
                throw runtime_error("Missing value after " + argument);
            }
            return string(argv[++index]);
        };

        if (argument == "--help" || argument == "-h") {
            print_help();
            exit(0);
        } else if (argument == "--input") {
            config.input = next_value();
        } else if (argument == "--features") {
            config.features = next_value();
        } else if (argument == "--synthetic") {
            config.synthetic = true;
        } else if (argument == "--points") {
            config.points = stoull(next_value());
        } else if (argument == "--k") {
            config.k = stoi(next_value());
        } else if (argument == "--threads") {
            config.threads = parse_threads(next_value());
        } else if (argument == "--repeats") {
            config.repeats = stoi(next_value());
        } else if (argument == "--max-iterations") {
            config.max_iterations = stoi(next_value());
        } else if (argument == "--tolerance") {
            config.tolerance = stod(next_value());
        } else if (argument == "--seed") {
            config.seed =
                static_cast<unsigned>(stoul(next_value()));
        } else if (argument == "--results") {
            config.results = next_value();
        } else if (argument == "--clusters") {
            config.clusters = next_value();
        } else if (argument == "--centroids") {
            config.centroids = next_value();
        } else {
            throw runtime_error("Unknown argument: " + argument);
        }
    }

    if (config.repeats < 1 || config.max_iterations < 1 ||
        config.tolerance < 0.0 || !isfinite(config.tolerance)) {
        throw runtime_error(
            "Invalid repeats, iteration limit, or tolerance.");
    }

    return config;
}

int main(int argc, char** argv) {
    try {
        const Config config = parse_arguments(argc, argv);
        const auto feature_names = split_names(config.features);
        if (feature_names.empty()) {
            throw runtime_error(
                "Select at least one numeric feature with --features.");
        }

        vector<Point> data =
            config.synthetic
                ? generate_synthetic_data(config.points, config.seed,
                                          feature_names.size())
                : read_customer_data(config.input, feature_names);

        if (config.k < 1 ||
            static_cast<size_t>(config.k) > data.size()) {
            throw runtime_error("K must be between 1 and row count.");
        }
        standardize_points(data);

        cout << "=== CST 435: Sequential vs OpenMP K-Means ===\n"
                  << "Rows: " << data.size() << " | Features: ";
        for (size_t index = 0; index < feature_names.size(); ++index) {
            cout << (index == 0 ? "" : ", ") << feature_names[index];
        }
        cout << "\nK: " << config.k
                  << " | Repeats: " << config.repeats << " | Threads: ";
        for (size_t index = 0; index < config.threads.size(); ++index) {
            cout << (index == 0 ? "" : ",") << config.threads[index];
        }

#ifdef _OPENMP
        cout << "\nOpenMP: enabled (runtime max threads="
                  << omp_get_max_threads() << ")\n";
#else
        cout
            << "\nOpenMP: DISABLED; enable OpenMP before using parallel "
               "results\n";
#endif
        cout
            << "Timing excludes CSV loading, standardization, centroid "
               "initialization and file writing.\n\n";

        vector<Measurement> measurements;
        Result final_result;
        const unsigned initialisation_seed = config.seed + 1000;

        for (int repeat = 1; repeat <= config.repeats; ++repeat) {
            const Result result =
                run_sequential(data, config.k, config.max_iterations,
                               config.tolerance, initialisation_seed);
            measurements.push_back(
                {"sequential", data.size(), config.k, 1, repeat,
                 result.iterations, result.elapsed_ms, result.wcss,
                 result.converged});

            if (repeat == config.repeats) {
                final_result = result;
            }

            cout << "Sequential run " << repeat << ": "
                      << fixed << setprecision(3)
                      << result.elapsed_ms << " ms; iterations="
                      << result.iterations << "; WCSS=" << result.wcss
                      << '\n';
        }

        for (const int thread_count : config.threads) {
            for (int repeat = 1; repeat <= config.repeats; ++repeat) {
                const Result result =
                    run_openmp(data, config.k, config.max_iterations,
                               config.tolerance, initialisation_seed,
                               thread_count);
                measurements.push_back(
                    {"openmp", data.size(), config.k, thread_count, repeat,
                     result.iterations, result.elapsed_ms, result.wcss,
                     result.converged});

                if (thread_count == config.threads.back() &&
                    repeat == config.repeats) {
                    final_result = result;
                }

                cout << "OpenMP " << thread_count << " thread(s), run "
                          << repeat << ": " << result.elapsed_ms
                          << " ms; iterations=" << result.iterations
                          << "; WCSS=" << result.wcss << '\n';
            }
        }

        save_measurements(config.results, measurements);
        save_cluster_assignments(config.clusters, final_result, feature_names);
        save_centers(config.centroids, final_result, feature_names);

        cout << "\nFinal exported clustering: WCSS="
                  << setprecision(10) << final_result.wcss
                  << "\nCluster sizes:";
        const auto cluster_sizes =
            calculate_cluster_sizes(final_result.points,
                                    final_result.centers.size());
        for (size_t cluster = 0; cluster < cluster_sizes.size();
             ++cluster) {
            cout << " cluster " << cluster << "="
                      << cluster_sizes[cluster];
        }
        cout << "\nSaved: " << config.results << ", "
                  << config.clusters << ", " << config.centroids << '\n';

        if (data.size() < 10000 && !config.synthetic) {
            cout
                << "NOTE: Mall Customers has only about 200 rows; use "
                   "--synthetic with larger --points for meaningful "
                   "scalability experiments.\n";
        }

        return 0;
    } catch (const exception& error) {
        cerr << "ERROR: " << error.what() << "\n\n";
        print_help();
        return 1;
    }
}

#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../cpp/include/graph.h"

template <typename T>
std::vector<T> parse_line_to_vector(const std::string& line) {
    std::istringstream iss(line);
    std::vector<T> vec;
    T val;
    while (iss >> val) vec.push_back(val);
    return vec;
}

inline std::vector<std::string> read_file_lines(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open test file: " + filepath);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    return lines;
}

inline std::string read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open test file: " + filepath);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

inline Graph parse_graph(const std::string& filepath) {
    auto lines = read_file_lines(filepath);
    int output_index = std::stoi(lines.back());
    lines.pop_back();

    std::vector<Node> nodes;
    for (size_t i = 0; i < lines.size(); i += 6) {
        Node n;
        n.op = static_cast<OpCode>(std::stoi(lines[i]));
        n.inputs = parse_line_to_vector<int>(lines[i + 1]);
        n.shape = parse_line_to_vector<int64_t>(lines[i + 2]);
        n.strides = parse_line_to_vector<int64_t>(lines[i + 3]);
        n.offset = std::stoll(lines[i + 4]);
        n.args = parse_line_to_vector<int64_t>(lines[i + 5]);
        nodes.push_back(n);
    }
    return Graph(std::move(nodes), output_index);
}

inline std::string dump_codegen(const Graph& g) {
    std::ostringstream o;
    for (const auto& c : g.configs) {
        o << (c.name.empty() ? "-" : c.name);
        for (auto x : c.grid) o << ' ' << x;
        for (auto x : c.group) o << ' ' << x;
        o << '\n';
    }
    o << "---\n";
    o << g.shader_source;
    return o.str();
}

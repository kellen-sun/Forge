#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <string>
#include <vector>

#include "../../cpp/include/common.h"
#include "graph_io.h"

#ifndef FORGE_SOURCE_DIR
#define FORGE_SOURCE_DIR ".."
#endif

static std::string repo_file(const char* rel) {
    return read_file(std::string(FORGE_SOURCE_DIR) + "/" + rel);
}

static std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

static std::string slice_between(const std::string& text, const std::string& start,
                                 const std::string& end) {
    const auto a = text.find(start);
    if (a == std::string::npos) return {};
    const auto b = text.find(end, a + start.size());
    if (b == std::string::npos) return {};
    return text.substr(a + start.size(), b - (a + start.size()));
}

static std::vector<std::string> quoted_strings_in_list(const std::string& text,
                                                       const std::string& marker, char open,
                                                       char close) {
    const auto mark = text.find(marker);
    if (mark == std::string::npos) return {};
    const auto begin = text.find(open, mark);
    if (begin == std::string::npos) return {};
    const auto end = text.find(close, begin + 1);
    if (end == std::string::npos) return {};
    const std::string region = text.substr(begin, end - begin);
    std::vector<std::string> names;
    for (size_t i = 0; i < region.size(); ++i) {
        if (region[i] != '"') continue;
        const auto j = region.find('"', i + 1);
        if (j == std::string::npos) break;
        names.push_back(region.substr(i + 1, j - i - 1));
        i = j;
    }
    return names;
}

static std::map<std::string, int> name_eq_ints(const std::string& region) {
    std::map<std::string, int> out;
    size_t i = 0;
    while (i < region.size()) {
        const auto nl = region.find('\n', i);
        std::string line = region.substr(i, (nl == std::string::npos ? region.size() : nl) - i);
        i = (nl == std::string::npos) ? region.size() : nl + 1;
        const auto comment = line.find("//");
        if (comment != std::string::npos) line = line.substr(0, comment);
        const auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (!line.empty() && line.back() == ',') line.pop_back();
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string name = trim(line.substr(0, eq));
        const std::string val = trim(line.substr(eq + 1));
        if (name.empty() || val.empty() || name == "COUNT") continue;
        bool ident = std::isalpha(static_cast<unsigned char>(name[0])) || name[0] == '_';
        for (char c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') ident = false;
        }
        if (!ident) continue;
        try {
            out.emplace(name, std::stoi(val));
        } catch (...) {
        }
    }
    return out;
}

static std::vector<std::string> metal_macro_names(const std::string& text, const std::string& macro) {
    const std::string def = "#define " + macro;
    const std::string call = macro + "(";
    std::vector<std::string> names;
    for (size_t pos = 0; (pos = text.find(call, pos)) != std::string::npos;) {
        size_t line_start = text.rfind('\n', pos);
        line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
        if (text.compare(line_start, def.size(), def) == 0) {
            pos += call.size();
            continue;
        }
        const auto comma = text.find(',', pos + call.size());
        if (comma == std::string::npos) break;
        names.push_back(trim(text.substr(pos + call.size(), comma - (pos + call.size()))));
        pos = comma;
    }
    return names;
}

static std::vector<std::string> make_binop_names(const std::string& ops_py) {
    const std::string call = "_make_binop(\"";
    std::vector<std::string> names;
    for (size_t pos = 0; (pos = ops_py.find(call, pos)) != std::string::npos;) {
        const auto end = ops_py.find('"', pos + call.size());
        if (end == std::string::npos) break;
        names.push_back(ops_py.substr(pos + call.size(), end - (pos + call.size())));
        pos = end;
    }
    return names;
}

TEST(PythonCppAbi, OpcodesMatch) {
    const auto py = name_eq_ints(slice_between(repo_file("py/Forge/graph.py"), "class Ops:", "class Node"));
    const auto cc = name_eq_ints(slice_between(repo_file("cpp/include/common.h"), "enum class OpCode", "COUNT"));
    ASSERT_FALSE(py.empty());
    EXPECT_EQ(py, cc) << "keep py/Forge/graph.py Ops in lockstep with cpp/include/common.h OpCode";
}

TEST(PythonCppAbi, UnaryNamesMatch) {
    const std::string ops_py = repo_file("py/Forge/ops.py");
    const std::string common_h = repo_file("cpp/include/common.h");
    const std::string metal = repo_file("cpp/include/metal_source.h");
    const auto from_py = quoted_strings_in_list(ops_py, "UNARY_OPS", '[', ']');
    const auto from_h = quoted_strings_in_list(common_h, "kUnaryNames", '{', '}');
    const auto from_metal = metal_macro_names(metal, "UNARY_OP");
    ASSERT_FALSE(from_py.empty());
    EXPECT_EQ(from_py, from_h) << "UNARY_OPS vs kUnaryNames";
    EXPECT_EQ(from_py, from_metal) << "UNARY_OPS vs METAL UNARY_OP kernels";
    ASSERT_EQ(static_cast<int>(from_h.size()), kUnaryCount);
    for (int i = 0; i < kUnaryCount; ++i) {
        EXPECT_STREQ(from_h[static_cast<size_t>(i)].c_str(), kUnaryNames[i]);
    }
}

TEST(PythonCppAbi, NullaryNamesMatch) {
    const auto from_py = quoted_strings_in_list(repo_file("py/Forge/ops.py"), "NULLARY_OPS", '[', ']');
    const auto from_metal = metal_macro_names(repo_file("cpp/include/metal_source.h"), "NULLARY_OP");
    ASSERT_FALSE(from_py.empty());
    EXPECT_EQ(from_py.front(), "rand");
    std::vector<std::string> py_gpu = from_py;
    py_gpu.erase(std::remove(py_gpu.begin(), py_gpu.end(), "zeros"), py_gpu.end());
    EXPECT_EQ(py_gpu, from_metal)
        << "NULLARY_OPS (minus zeros, which is a CPU fill) vs METAL NULLARY_OP kernels";
    EXPECT_NE(std::find(from_py.begin(), from_py.end(), "zeros"), from_py.end());
}

TEST(PythonCppAbi, BinaryAndInplaceKernelNamesMatch) {
    const std::string ops_py = repo_file("py/Forge/ops.py");
    const std::string metal = repo_file("cpp/include/metal_source.h");
    const auto py_bin = make_binop_names(ops_py);
    std::vector<std::string> py_binary, py_inplace;
    for (const auto& n : py_bin) {
        if (!n.empty() && n[0] == 'i')
            py_inplace.push_back(n);
        else
            py_binary.push_back(n);
    }
    EXPECT_EQ(py_binary, metal_macro_names(metal, "BINARY_OP"));
    EXPECT_EQ(py_inplace, metal_macro_names(metal, "INPLACE_OP"));
}

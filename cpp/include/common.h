#pragma once
#include <map>

enum class OpCode : int {
    INPUT = 0,
    MATMUL = 1,
    ADD = 2,
    MUL = 3,
    DIV = 4,
    SUB = 5,
    RESHAPE = 6,
    TRANSPOSE = 7,
    VIEW = 8,
    UPDATE = 9,
    CONSTANT = 10,
    COPY = 11
};

const std::map<OpCode, std::string> op_symbol = {
    {OpCode::ADD, "+"},
    {OpCode::SUB, "-"},
    {OpCode::MUL, "*"},
    {OpCode::DIV, "/"},
};

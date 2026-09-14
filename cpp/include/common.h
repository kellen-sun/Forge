#pragma once

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
    COPY = 11,
    SUM = 12,
    // Add new ops before COUNT which is a sentinel for the number of opcodes
    COUNT
};

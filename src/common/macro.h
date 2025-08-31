#pragma once

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

template <typename T>
using GemmType = void (*)(T*, T*, T*, const int, const int, const int);


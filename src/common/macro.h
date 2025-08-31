#pragma once

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

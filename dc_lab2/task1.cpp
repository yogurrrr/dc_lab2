#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>

std::vector<int> matrixVectorMultiplyByRows(const std::vector<std::vector<int>>& matrix, const std::vector<int>& vector) {
    int size = matrix.size();
    std::vector<int> result(size, 0);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int blockSize = size / numProcs;
    int startRow = rank * blockSize;
    int endRow = (rank == numProcs - 1) ? size : startRow + blockSize;

    std::vector<int> localResult(size, 0);

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < size; ++j) {
            localResult[i] += matrix[i][j] * vector[j];
        }
    }

    MPI_Reduce(localResult.data(), result.data(), size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    return result;
}

std::vector<int> matrixVectorMultiplyByColumns(const std::vector<std::vector<int>>& matrix, const std::vector<int>& vector) {
    int size = matrix.size();
    std::vector<int> result(size, 0);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int columnsPerProc = size / numProcs;
    int startColumn = rank * columnsPerProc;
    int endColumn = (rank == numProcs - 1) ? size : startColumn + columnsPerProc;

    for (int j = startColumn; j < endColumn; ++j) {
        for (int i = 0; i < size; ++i) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    return result;
}


std::vector<int> matrixVectorMultiply(const std::vector<std::vector<int>>& matrix, const std::vector<int>& vector) {
    int size = matrix.size();
    std::vector<int> result(size, 0);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int blockSize = size / numProcs;
    int startRow = rank * blockSize;
    int endRow = (rank == numProcs - 1) ? size : startRow + blockSize;

    std::vector<int> localResult(blockSize, 0);

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < size; ++j) {
            localResult[i - startRow] += matrix[i][j] * vector[j];
        }
    }

    MPI_Gather(localResult.data(), blockSize, MPI_INT, result.data(), blockSize, MPI_INT, 0, MPI_COMM_WORLD);

    return result;
}

int main() {
    MPI_Init(NULL, NULL);

    int size = 16; 
    srand(time(NULL));

    std::vector<std::vector<int>> matrix(size, std::vector<int>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = rand() % 10;
        }
    }

    std::vector<int> vector(size);
    for (int i = 0; i < size; ++i) {
        vector[i] = rand() % 10;
    }

    std::vector<int> result;

    double startTime = MPI_Wtime();

    //result = matrixVectorMultiply(matrix, vector);
    //result = matrixVectorMultiplyByRows(matrix, vector);
    result = matrixVectorMultiplyByColumns(matrix, vector);

    double endTime = MPI_Wtime();
    double elapsedTime = endTime - startTime;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "Original Matrix:" << std::endl;
        for (const auto& row : matrix) {
            for (int val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Original Vector:" << std::endl;
        for (int val : vector) {
            std::cout << val << "\n";
        }
        std::cout << std::endl;

        std::cout << "Result Vector:" << std::endl;
        for (int val : result) {
            std::cout << val << "\n";
        }
        std::cout << std::endl;

        std::cout << "Elapsed Time: " << elapsedTime << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}


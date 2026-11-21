#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4 // Number of local interior compute cells

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This demo requires exactly 2 MPI processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Allocate array of size N + 1 (N interior cells + 1 ghost cell)
    int arr[N + 1];
    int partner = 1 - rank; // If rank is 0, partner is 1; if rank is 1, partner is 0
    MPI_Request reqs[2];     // reqs[0] for recv, reqs[1] for send

    // 1. Initialize local data
    if (rank == 0) {
        // Rank 0 owns cells 0 to N-1; cell N is the ghost cell
        for (int i = 0; i < N; i++) {
            arr[i] = (i + 1) * 10; // e.g., 10, 20, 30, 40
        }
        arr[N] = -1; // Uninitialized ghost cell
    } else {
        // Rank 1 owns cells 1 to N; cell 0 is the ghost cell
        arr[0] = -1; // Uninitialized ghost cell
        for (int i = 1; i <= N; i++) {
            arr[i] = (i + 1) * 100; // e.g., 200, 300, 400, 500
        }
    }

    // 2. Post Non-Blocking Receive and Send (Halo Exchange)
    if (rank == 0) {
        // Receive neighbor's edge into right ghost cell arr[N]
        MPI_Irecv(&arr[N], 1, MPI_INT, partner, 0, MPI_COMM_WORLD, &reqs[0]);
        // Send our rightmost boundary cell arr[N-1] to Rank 1
        MPI_Isend(&arr[N - 1], 1, MPI_INT, partner, 0, MPI_COMM_WORLD, &reqs[1]);
    } else {
        // Receive neighbor's edge into left ghost cell arr[0]
        MPI_Irecv(&arr[0], 1, MPI_INT, partner, 0, MPI_COMM_WORLD, &reqs[0]);
        // Send our leftmost boundary cell arr[1] to Rank 0
        MPI_Isend(&arr[1], 1, MPI_INT, partner, 0, MPI_COMM_WORLD, &reqs[1]);
    }

    // 3. Optional: Perform independent interior computations here to overlap with network transfer!

    // 4. Synchronize: Wait until both non-blocking operations are complete
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    // 5. Verify the updated ghost cell
    if (rank == 0) {
        printf("Rank 0: arr[N-1] (sent) = %d | arr[N] (ghost received) = %d\n",
               arr[N - 1], arr[N]);
    } else {
        printf("Rank 1: arr[1] (sent) = %d | arr[0] (ghost received) = %d\n",
               arr[1], arr[0]);
    }

    MPI_Finalize();
    return 0;
}
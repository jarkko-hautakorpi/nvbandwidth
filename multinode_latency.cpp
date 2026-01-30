/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Multinode Latency Tests for nvbandwidth
 * Implements GPU-to-GPU latency measurement across nodes using IMEX
 */

#ifdef MULTINODE

#include "testcase.h"
#include "kernels.cuh"
#include "memcpy.h"
#include "common.h"
#include "output.h"
#include "multinode_memcpy.h"
#include <mpi.h>

// Multinode Device-to-Device Latency using pointer chase
void MultinodeDeviceToDeviceLatencySM::run(unsigned long long size, unsigned long long loopCount) {
    // Create matrix to store latency values across all nodes
    // Each MPI rank contributes its local GPU latencies to remote GPUs
    int totalGPUs = worldSize;
    PeerValueMatrix<double> latencyValues(totalGPUs, totalGPUs, key, perfFormatter, LATENCY);
    
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);
    
    // Each rank tests from its local GPU to all remote GPUs
    for (int targetRank = 0; targetRank < worldSize; targetRank++) {
        // Allocate buffer on target rank's GPU (using multinode buffer)
        MultinodeDeviceBuffer targetBuffer(size, targetRank);
        
        // Initialize the buffer for pointer chasing
        if (worldRank == targetRank) {
            latencyHelper(targetBuffer, true);
        }
        
        // Synchronize to ensure buffer is initialized
        MPI_Barrier(MPI_COMM_WORLD);
        
        for (int srcRank = 0; srcRank < worldSize; srcRank++) {
            if (srcRank == targetRank) {
                // Same-node latency is not measured in multinode test
                continue;
            }
            
            // Only the source rank performs the measurement
            if (worldRank == srcRank) {
                // Perform pointer chase from srcRank's GPU to targetRank's GPU
                double latency = ptrChaseOp.doPtrChase(localDevice, targetBuffer);
                latencyValues.value(srcRank, targetRank) = latency;
            }
        }
        
        // Synchronize after each target to avoid conflicts
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Gather all latency values to rank 0 for output
    if (worldRank == 0) {
        // Rank 0 already has its own measurements
        // Receive measurements from other ranks
        for (int srcRank = 1; srcRank < worldSize; srcRank++) {
            for (int targetRank = 0; targetRank < worldSize; targetRank++) {
                if (srcRank != targetRank) {
                    double latency;
                    MPI_Recv(&latency, 1, MPI_DOUBLE, srcRank, 
                            srcRank * worldSize + targetRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    latencyValues.value(srcRank, targetRank) = latency;
                }
            }
        }
    } else {
        // Send measurements to rank 0
        for (int targetRank = 0; targetRank < worldSize; targetRank++) {
            if (worldRank != targetRank) {
                // Extract value from optional
                auto latency_opt = latencyValues.value(worldRank, targetRank);
                if (latency_opt.has_value()) {
                    double latency = latency_opt.value();
                    MPI_Send(&latency, 1, MPI_DOUBLE, 0, 
                            worldRank * worldSize + targetRank, MPI_COMM_WORLD);
                }
            }
        }
    }
    
    // Output results (only rank 0)
    if (worldRank == 0) {
        output->addTestcaseResults(latencyValues, 
            "Multinode Device to Device Latency SM GPU(row) <-> GPU(column) across nodes (ns)");
    }
}

// Multinode Host-to-Device Latency
void MultinodeHostDeviceLatencySM::run(unsigned long long size, unsigned long long loopCount) {
    // Measure latency from local CPU to remote GPU
    int totalGPUs = worldSize;
    PeerValueMatrix<double> latencyValues(1, totalGPUs, key, perfFormatter, LATENCY);
    
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);
    
    for (int targetRank = 0; targetRank < worldSize; targetRank++) {
        // Allocate buffer on target rank's GPU
        MultinodeDeviceBuffer targetBuffer(size, targetRank);
        
        // Initialize buffer on target
        if (worldRank == targetRank) {
            latencyHelper(targetBuffer, false);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Rank 0 measures latency to all GPUs
        if (worldRank == 0) {
            double latency = ptrChaseOp.doPtrChase(localDevice, targetBuffer);
            latencyValues.value(0, targetRank) = latency;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (worldRank == 0) {
        output->addTestcaseResults(latencyValues, 
            "Multinode Host to Device Latency SM CPU(row) <-> GPU(column) across nodes (ns)");
    }
}

// Multinode Bidirectional Latency (measures round-trip)
void MultinodeDeviceToDeviceBidirLatencySM::run(unsigned long long size, unsigned long long loopCount) {
    int totalGPUs = worldSize;
    PeerValueMatrix<double> latencyValuesForward(totalGPUs, totalGPUs, key + "_forward", perfFormatter, LATENCY);
    PeerValueMatrix<double> latencyValuesReverse(totalGPUs, totalGPUs, key + "_reverse", perfFormatter, LATENCY);
    PeerValueMatrix<double> latencyValuesRTT(totalGPUs, totalGPUs, key + "_rtt", perfFormatter, LATENCY);
    
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);
    
    // Measure forward and reverse latency
    for (int pairIdx = 0; pairIdx < worldSize - 1; pairIdx++) {
        int rank1 = pairIdx;
        int rank2 = pairIdx + 1;
        
        // Allocate buffers on both ranks
        MultinodeDeviceBuffer buffer1(size, rank1);
        MultinodeDeviceBuffer buffer2(size, rank2);
        
        // Initialize buffers
        if (worldRank == rank1 || worldRank == rank2) {
            if (worldRank == rank1) latencyHelper(buffer1, true);
            if (worldRank == rank2) latencyHelper(buffer2, true);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Measure rank1 -> rank2
        if (worldRank == rank1) {
            double latency = ptrChaseOp.doPtrChase(localDevice, buffer2);
            latencyValuesForward.value(rank1, rank2) = latency;
        }
        
        // Measure rank2 -> rank1
        if (worldRank == rank2) {
            double latency = ptrChaseOp.doPtrChase(localDevice, buffer1);
            latencyValuesReverse.value(rank2, rank1) = latency;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Gather results to rank 0
    if (worldRank == 0) {
        for (int pairIdx = 0; pairIdx < worldSize - 1; pairIdx++) {
            int rank1 = pairIdx;
            int rank2 = pairIdx + 1;
            
            // Receive from rank1
            if (rank1 != 0) {
                double latency;
                MPI_Recv(&latency, 1, MPI_DOUBLE, rank1, rank1 * 1000 + rank2, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                latencyValuesForward.value(rank1, rank2) = latency;
            }
            
            // Receive from rank2
            if (rank2 != 0) {
                double latency;
                MPI_Recv(&latency, 1, MPI_DOUBLE, rank2, rank2 * 1000 + rank1, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                latencyValuesReverse.value(rank2, rank1) = latency;
            }
            
            // Calculate RTT - extract values from optional
            auto fwd_opt = latencyValuesForward.value(rank1, rank2);
            auto rev_opt = latencyValuesReverse.value(rank2, rank1);
            
            if (fwd_opt.has_value() && rev_opt.has_value()) {
                double fwd = fwd_opt.value();
                double rev = rev_opt.value();
                latencyValuesRTT.value(rank1, rank2) = fwd + rev;
                latencyValuesRTT.value(rank2, rank1) = fwd + rev;
            }
        }
        
        output->addTestcaseResults(latencyValuesForward, 
            "Multinode Device to Device Forward Latency SM (ns)");
        output->addTestcaseResults(latencyValuesReverse, 
            "Multinode Device to Device Reverse Latency SM (ns)");
        output->addTestcaseResults(latencyValuesRTT, 
            "Multinode Device to Device Round-Trip Latency SM (ns)");
    } else {
        // Send measurements to rank 0
        for (int pairIdx = 0; pairIdx < worldSize - 1; pairIdx++) {
            int rank1 = pairIdx;
            int rank2 = pairIdx + 1;
            
            if (worldRank == rank1) {
                auto latency_opt = latencyValuesForward.value(rank1, rank2);
                if (latency_opt.has_value()) {
                    double latency = latency_opt.value();
                    MPI_Send(&latency, 1, MPI_DOUBLE, 0, rank1 * 1000 + rank2, MPI_COMM_WORLD);
                }
            }
            
            if (worldRank == rank2) {
                auto latency_opt = latencyValuesReverse.value(rank2, rank1);
                if (latency_opt.has_value()) {
                    double latency = latency_opt.value();
                    MPI_Send(&latency, 1, MPI_DOUBLE, 0, rank2 * 1000 + rank1, MPI_COMM_WORLD);
                }
            }
        }
    }
}

#endif // MULTINODE


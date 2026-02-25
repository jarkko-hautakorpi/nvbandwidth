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

// Helper function to initialize multinode buffer for latency tests
static void initializeMultinodeLatencyBuffer(MultinodeDeviceBuffer& buffer, size_t bufferSize, bool deviceMemory) {
    // Only the owner rank initializes the buffer
    if (worldRank != buffer.getMPIRank()) {
        return;
    }
    
    size_t numNodes = bufferSize / sizeof(struct LatencyNode);
    
    CUdeviceptr dataBuffer = (CUdeviceptr)buffer.getBuffer();
    
    // Set context for this device
    CUcontext ctx = buffer.getPrimaryCtx();
    CU_ASSERT(cuCtxSetCurrent(ctx));
    
    // Initialize linked list for pointer chasing
    for (size_t i = 0; i < numNodes; i++) {
        struct LatencyNode node;
        // Create a chain: node[i] -> node[i+1] -> ... -> node[n-1] -> node[0]
        node.next = (struct LatencyNode*)(dataBuffer + ((i + 1) % numNodes) * sizeof(struct LatencyNode));
        
        // Copy from host to device
        CU_ASSERT(cuMemcpyHtoD(dataBuffer + i * sizeof(struct LatencyNode), 
                              &node, sizeof(struct LatencyNode)));
    }
    
    // Release context
    CU_ASSERT(cuDevicePrimaryCtxRelease(localDevice));
}

// Multinode Device-to-Device Latency using pointer chase
void MultinodeDeviceToDeviceLatencySM::run(unsigned long long size, unsigned long long loopCount) {
    // Create matrix to store latency values across all nodes
    int totalGPUs = worldSize;
    PeerValueMatrix<double> latencyValues(totalGPUs, totalGPUs, key, perfFormatter, LATENCY);
    
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);
    
    // Each rank tests from its local GPU to all remote GPUs
    for (int targetRank = 0; targetRank < worldSize; targetRank++) {
        // Allocate buffer on target rank's GPU
        MultinodeDeviceBufferUnicast targetBuffer(size, targetRank);
        
        // Initialize the buffer for pointer chasing (only on owner rank)
        initializeMultinodeLatencyBuffer(targetBuffer, size, true);
        
        // Synchronize to ensure buffer is initialized
        MPI_Barrier(MPI_COMM_WORLD);
        
        for (int srcRank = 0; srcRank < worldSize; srcRank++) {
            if (srcRank == targetRank) {
                // Skip same-node latency in multinode test
                continue;
            }
            
            // Only the source rank performs the measurement
            if (worldRank == srcRank) {
                // Set current device context
                CUcontext ctx;
                CU_ASSERT(cuDevicePrimaryCtxRetain(&ctx, localDevice));
                CU_ASSERT(cuCtxSetCurrent(ctx));
                
                // Perform pointer chase from srcRank's GPU to targetRank's GPU
                double latency = ptrChaseOp.doPtrChase(localDevice, targetBuffer);
                latencyValues.value(srcRank, targetRank) = latency;
                
                CU_ASSERT(cuDevicePrimaryCtxRelease(localDevice));
            }
        }
        
        // Synchronize after each target to avoid conflicts
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Gather all latency values to rank 0 for output
    if (worldRank == 0) {
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
        MultinodeDeviceBufferUnicast targetBuffer(size, targetRank);
        
        // Initialize buffer on target (host-accessible memory)
        initializeMultinodeLatencyBuffer(targetBuffer, size, false);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Rank 0 measures latency to all GPUs
        if (worldRank == 0) {
            CUcontext ctx;
            CU_ASSERT(cuDevicePrimaryCtxRetain(&ctx, localDevice));
            CU_ASSERT(cuCtxSetCurrent(ctx));
            
            double latency = ptrChaseOp.doPtrChase(localDevice, targetBuffer);
            latencyValues.value(0, targetRank) = latency;
            
            CU_ASSERT(cuDevicePrimaryCtxRelease(localDevice));
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
        MultinodeDeviceBufferUnicast buffer1(size, rank1);
        MultinodeDeviceBufferUnicast buffer2(size, rank2);
        
        // Initialize buffers
        initializeMultinodeLatencyBuffer(buffer1, size, true);
        initializeMultinodeLatencyBuffer(buffer2, size, true);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Measure rank1 -> rank2
        if (worldRank == rank1) {
            CUcontext ctx;
            CU_ASSERT(cuDevicePrimaryCtxRetain(&ctx, localDevice));
            CU_ASSERT(cuCtxSetCurrent(ctx));
            
            double latency = ptrChaseOp.doPtrChase(localDevice, buffer2);
            latencyValuesForward.value(rank1, rank2) = latency;
            
            CU_ASSERT(cuDevicePrimaryCtxRelease(localDevice));
        }
        
        // Measure rank2 -> rank1
        if (worldRank == rank2) {
            CUcontext ctx;
            CU_ASSERT(cuDevicePrimaryCtxRetain(&ctx, localDevice));
            CU_ASSERT(cuCtxSetCurrent(ctx));
            
            double latency = ptrChaseOp.doPtrChase(localDevice, buffer1);
            latencyValuesReverse.value(rank2, rank1) = latency;
            
            CU_ASSERT(cuDevicePrimaryCtxRelease(localDevice));
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
            
            // Calculate RTT
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


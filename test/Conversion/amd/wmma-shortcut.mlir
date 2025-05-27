// RUN: triton-opt %s --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx1100" -split-input-file | FileCheck %s

#wmma = #ttg.amd_wmma<{version = 1, warpsPerCTA = [2, 1], isTransposed = false}>
#wmmaT = #ttg.amd_wmma<{version = 1, warpsPerCTA = [2, 1], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #wmma, kWidth=16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot_cvt_bf16_wmma
  tt.func public @wmma_dot_cvt_bf16_wmma(%arg0: tensor<32x32xbf16, #wmmaT>) {
    // CHECK-NOT: store
    // CHECK-NOT: load

    // CHECK: [[val0:%.*]] = llvm.extractvalue %arg0[0]
    // CHECK: [[c32:%.*]] = llvm.mlir.constant(32 : i32)
    // CHECK: [[c16:%.*]] = llvm.mlir.constant(16 : i32)
    // CHECK: [[workitem:%.*]] = rocdl.workitem.id.x
    // CHECK: [[laneId:%.*]] = llvm.urem [[workitem]], [[c32]]
    // CHECK: [[isLower:%.*]] = llvm.icmp "slt" [[laneId]], [[c16]]

    // CHECK: [[val0I:%.*]] = llvm.bitcast [[val0]]
    // CHECK: [[selectLo:%.*]] = llvm.mlir.constant(1985229328 : i32)
    // CHECK: [[selectHi:%.*]] = llvm.mlir.constant(-19088744 : i32)
    // CHECK: [[fetchInactive:%.*]] = llvm.mlir.constant(true)
    // CHECK: [[boundControl:%.*]] = llvm.mlir.constant(false)
    // CHECK: [[val0ISwapped:%.*]] = llvm.call_intrinsic "llvm.amdgcn.permlanex16"([[val0I]], [[val0I]], [[selectLo]], [[selectHi]], [[fetchInactive]], [[boundControl]]) : (i16, i16, i32, i32, i1, i1) -> i16
    // CHECK: [[val0Swapped:%.*]] = llvm.bitcast [[val0ISwapped]] : i16 to bf16
    // CHECK: [[res0:%.*]] = llvm.select [[isLower]], [[val0]], [[val0Swapped]]
    // CHECK: [[res1:%.*]] = llvm.select [[isLower]], [[val0Swapped]], [[val0]]

    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<32x32xbf16, #wmmaT> -> tensor<32x32xbf16, #dotop0>
    tt.return
  }
}
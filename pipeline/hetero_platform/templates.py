# SPDX-License-Identifier: Apache-2.0
"""Templates that emit a cluster offload instead of a direct kernel call.

Each mirrors the Deeploy Generic template it replaces -- same batch loop, same
pointer arithmetic, same operatorRepresentation keys -- and swaps the kernel
call for the matching hes_offload_* wrapper with an engine id in front. Keeping
the surrounding structure identical is deliberate: the generated network stays
readable next to the Generic one, and a node that moves between engines differs
only in the call.

BEGIN_SINGLE_CORE / END_SINGLE_CORE are the Generic platform's markers and
expand to nothing here; the host is a single core.
"""

from Deeploy.DeeployTypes import NodeTemplate


def matmul(engine: str) -> NodeTemplate:
    return NodeTemplate(f"""
// MatMul on {engine} (Name: ${{nodeName}}, Op: ${{nodeOp}})
BEGIN_SINGLE_CORE
    ${{A_type.typeName}} ref_${{data_out}}_${{A}} = ${{A}};
    ${{B_type.typeName}} ref_${{data_out}}_${{B}} = ${{B}};
    ${{data_out_type.typeName}} ref_${{data_out}}_${{data_out}} = ${{data_out}};

    for(uint32_t i=0; i<${{batch}}; i++){{
        hes_offload_matmul({engine},
            ref_${{data_out}}_${{A}},
            ref_${{data_out}}_${{B}},
            ref_${{data_out}}_${{data_out}},
            ${{M}}, ${{N}}, ${{O}});

        ref_${{data_out}}_${{A}} += ${{M}} * ${{N}};
        ref_${{data_out}}_${{B}} += ${{N}} * ${{O}};
        ref_${{data_out}}_${{data_out}} += ${{M}} * ${{O}};
    }}
END_SINGLE_CORE
""")


def gemm(engine: str) -> NodeTemplate:
    return NodeTemplate(f"""
// GEMM on {engine} (Name: ${{nodeName}}, Op: ${{nodeOp}})
BEGIN_SINGLE_CORE
    ${{A_type.typeName}} ref_${{data_out}}_${{A}} = ${{A}};
    ${{B_type.typeName}} ref_${{data_out}}_${{B}} = ${{B}};
    ${{C_type.typeName}} ref_${{data_out}}_${{C}} = ${{C}};
    ${{data_out_type.typeName}} ref_${{data_out}}_${{data_out}} = ${{data_out}};

    for(uint32_t i=0; i<${{batch}}; i++){{
        hes_offload_gemm({engine},
            ref_${{data_out}}_${{A}},
            ref_${{data_out}}_${{B}},
            ref_${{data_out}}_${{C}},
            ref_${{data_out}}_${{data_out}},
            ${{M}}, ${{N}}, ${{O}},
            ${{transA}}, ${{transB}});

        % if A_batched:
        ref_${{data_out}}_${{A}} += ${{M}} * ${{N}};
        % endif
        % if B_batched:
        ref_${{data_out}}_${{B}} += ${{N}} * ${{O}};
        % endif
        % if C_batched:
        ref_${{data_out}}_${{C}} += ${{M}} * ${{O}};
        % endif
        ref_${{data_out}}_${{data_out}} += ${{M}} * ${{O}};
    }}
END_SINGLE_CORE
""")


def conv2d(engine: str) -> NodeTemplate:
    return NodeTemplate(f"""
<%
batchOffsetIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>

// 2D FP Conv on {engine} (Name: ${{nodeName}}, Op: ${{nodeOp}})
BEGIN_SINGLE_CORE
    ${{data_in_type.typeName}} ref_${{data_out}}_${{data_in}} = ${{data_in}};
    ${{data_out_type.typeName}} ref_${{data_out}}_${{data_out}} = ${{data_out}};

    for (uint32_t n=0; n<${{batch}}; ++n) {{
        hes_offload_conv2d({engine},
            ref_${{data_out}}_${{data_in}}, ${{ch_im_in}}, ${{dim_im_in_x}}, ${{dim_im_in_y}},
            ${{weight}}, ${{ch_im_out}}, ${{dim_kernel_x}}, ${{dim_kernel_y}},
            ${{stride_x}}, ${{stride_y}},
            ${{bias}}, ${{has_bias}},
            ref_${{data_out}}_${{data_out}});

        ref_${{data_out}}_${{data_in}} += ${{batchOffsetIn}};
        ref_${{data_out}}_${{data_out}} += ${{batchOffsetOut}};
    }}
END_SINGLE_CORE
""")

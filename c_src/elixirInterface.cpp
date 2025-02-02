#include <erl_nif.h>
#include <cuda_fp16.h>
#include "ops.cuh"

void quantizeBlockwise_fp16_fp4(float* code, half* A, float* absmax, unsigned char* out, int blocksize, const int n) {
    quantizeBlockwise<half, 0, FP4>(NULL, A, absmax, out, NULL, 0, blocksize, n);
}

static ERL_NIF_TERM do_cquantize_blockwise_fp16_fp4(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary code_bin, tensor_bin, absmax_bin, out_bin;
    int blocksize, n;
    
    if (!enif_inspect_binary(env, argv[0], &code_bin) ||
        !enif_inspect_binary(env, argv[1], &tensor_bin) ||
        !enif_inspect_binary(env, argv[2], &absmax_bin) ||
        !enif_inspect_binary(env, argv[3], &out_bin) ||
        !enif_get_int(env, argv[4], &blocksize) ||
        !enif_get_int(env, argv[5], &n)) {
        return enif_make_badarg(env);
    }

    quantizeBlockwise_fp16_fp4(
        reinterpret_cast<float*>(code_bin.data),
        reinterpret_cast<half*>(tensor_bin.data),
        reinterpret_cast<float*>(absmax_bin.data),
        reinterpret_cast<unsigned char*>(out_bin.data),
        blocksize,
        n
    );

    return enif_make_binary(env, &out_bin);
}

static ErlNifFunc nif_funcs[] = {
    {"do_cquantize_blockwise_fp16_fp4", 6, do_cquantize_blockwise_fp16_fp4, 0}
};

ERL_NIF_INIT(Elixir.BitsAndBytes.NIF, nif_funcs, NULL, NULL, NULL, NULL)

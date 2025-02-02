defmodule BitsAndBytesTest do
  use ExUnit.Case
  import Bitwise

  describe "quantize_4bit/2" do
    test "basic quantization size check" do
      key = Nx.Random.key(123)
      blocksize = 64
      tensor = Nx.Random.uniform(key, shape: {64, 64}, type: {:f, 16}) |> elem(0)

      {quantized, state} = BitsAndBytes.quantize_4bit(tensor,
        blocksize: blocksize,
        quant_type: "fp4"
      )

      assert quantized.shape == {div(64 * 64 + 1, 2)}
      assert state.absmax.shape == {div(64 * 64 + blocksize - 1, blocksize)}
    end

    test "validates that output values are 4-bit (between 0 and 15)" do
      key = Nx.Random.key(123)
      tensor = Nx.Random.uniform(key, shape: {64, 64}, type: {:f, 16}) |> elem(0)

      {quantized, _state} = BitsAndBytes.quantize_4bit(tensor)

      binary = Nx.to_binary(quantized)
      unpacked_values = for <<byte <- binary>>, into: [] do
        [(byte >>> 4) &&& 0x0F, byte &&& 0x0F]
      end |> List.flatten()

      assert Enum.all?(unpacked_values, &(&1 >= 0 and &1 <= 15))
    end

    test "handles different block sizes" do
      key = Nx.Random.key(123)
      tensor = Nx.Random.uniform(key, shape: {128, 128}, type: {:f, 16}) |> elem(0)

      blocksizes = [64, 128, 256]

      results = for blocksize <- blocksizes do
        {_quantized, state} = BitsAndBytes.quantize_4bit(tensor, blocksize: blocksize)
        num_blocks = div(128 * 128 + blocksize - 1, blocksize)
        assert state.absmax.shape == {num_blocks}
      end

      assert length(results) == length(blocksizes)
    end

    test "quantization preserves tensor shape info" do
      original_shape = {32, 48}
      key = Nx.Random.key(123)
      tensor = Nx.Random.uniform(key, shape: original_shape, type: {:f, 16}) |> elem(0)

      {_quantized, state} = BitsAndBytes.quantize_4bit(tensor)

      assert state.shape == original_shape
      assert state.dtype == {:f, 16}
    end

    test "rejects incorrect input types" do
      key = Nx.Random.key(123)
      tensor = Nx.Random.uniform(key, shape: {64, 64}, type: {:f, 32}) |> elem(0)

      assert_raise RuntimeError, ~r/Blockwise quantization only supports/, fn ->
        BitsAndBytes.quantize_4bit(tensor)
      end
    end

    test "provides correct quantization type in state" do
      key = Nx.Random.key(123)
      tensor = Nx.Random.uniform(key, shape: {64, 64}, type: {:f, 16}) |> elem(0)

      {_quantized, state} = BitsAndBytes.quantize_4bit(tensor, quant_type: "fp4")

      assert state.quant_type == "fp4"
    end

    test "code tensor has correct shape and values" do
      key = Nx.Random.key(123)
      tensor = Nx.Random.uniform(key, shape: {64, 64}, type: {:f, 16}) |> elem(0)

      {_quantized, state} = BitsAndBytes.quantize_4bit(tensor)

      assert state.code.shape == {16}
      assert state.code.type == {:f, 32}

      assert Nx.to_number(state.code[0]) == 0.0
    end

    test "quantization is deterministic" do
      key = Nx.Random.key(123)
      tensor = Nx.Random.uniform(key, shape: {64, 64}, type: {:f, 16}) |> elem(0)

      {quantized1, _} = BitsAndBytes.quantize_4bit(tensor)
      {quantized2, _} = BitsAndBytes.quantize_4bit(tensor)

      assert Nx.to_binary(quantized1) == Nx.to_binary(quantized2)
    end
  end
end

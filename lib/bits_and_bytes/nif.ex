defmodule BitsAndBytes.NIF do
  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:bits_and_bytes), "libbits_and_bytes")
    case :erlang.load_nif(path, 0) do
      :ok -> :ok
      {:error, reason} -> raise "Failed to load NIF: #{reason}"
    end
  end

  def cquantize_blockwise_fp16_fp4(code, tensor, absmax, out, blocksize, n) do
    case do_cquantize_blockwise_fp16_fp4(
      Nx.to_binary(code),
      Nx.to_binary(tensor),
      Nx.to_binary(absmax),
      Nx.to_binary(out),
      blocksize,
      n
    ) do
      :error -> raise "Quantization failed"
      out_bin -> Nx.from_binary(out_bin, {:u, 8})
    end
  end

  defp do_cquantize_blockwise_fp16_fp4(_code, _tensor, _absmax, _out, _blocksize, _n) do
    :erlang.nif_error(:nif_not_loaded)
  end
end

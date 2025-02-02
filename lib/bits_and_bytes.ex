defmodule BitsAndBytes do
  @moduledoc """
  Documentation for `BitsAndBytes`.
  """

  def quantize_4bit(tensor, opts \\ []) do
    blocksize = Keyword.get(opts, :blocksize, 64)
    quant_type = Keyword.get(opts, :quant_type, "fp4")

    unless tensor.type == {:f, 16} do
      raise "Blockwise quantization only supports 16/32-bit floats, but got #{inspect(tensor.type)}"
    end

    unless blocksize in [4096, 2048, 1024, 512, 256, 128, 64] do
      raise "Blocksize must be one of [4096, 2048, 1024, 512, 256, 128, 64], but got #{blocksize}"
    end

    n = Nx.size(tensor)
    blocks = div(n + blocksize - 1, blocksize)
    absmax = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {blocks})
    out = Nx.broadcast(Nx.tensor(0, type: {:u, 8}), {div(n + 1, 2)})
    code = get_4bit_type(quant_type)

    quantized = BitsAndBytes.NIF.cquantize_blockwise_fp16_fp4(
      code,
      tensor,
      absmax,
      out,
      blocksize,
      n
    )

    {quantized, %{
      absmax: absmax,
      shape: tensor.shape,
      dtype: tensor.type,
      blocksize: blocksize,
      code: code,
      quant_type: quant_type
    }}
  end

  defp get_4bit_type("fp4") do
    data = [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0,
            -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
    tensor = Nx.tensor(data, type: {:f, 32})
    max_abs = Nx.abs(tensor) |> Nx.reduce_max()
    Nx.divide(tensor, max_abs)
  end
end

defmodule BitsAndBytes.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :bits_and_bytes,
      version: @version,
      elixir: "~> 1.14",
      compilers: [:cmake] ++ Mix.compilers(),
      deps: deps(),
      aliases: aliases()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.5"},
      {:torchx, "~> 0.5"}
    ]
  end

  defp aliases do
    [
      "compile.cmake": &cmake/1
    ]
  end

  def cmake(_args) do
    Mix.Task.run("deps.compile", ["torchx", "--force"])

    cmake = System.find_executable("cmake") || Mix.raise("cmake not found in the path")
    cmake_build_dir = Path.join(Mix.Project.app_path(), "cmake")
    File.mkdir_p!(cmake_build_dir)

    erts_include_dir =
      Path.join([:code.root_dir(), "erts-#{:erlang.system_info(:version)}", "include"])

    env = %{
      "MIX_BUILD_EMBEDDED" => "#{Mix.Project.config()[:build_embedded]}",
      "ERTS_INCLUDE_DIR" => erts_include_dir
    }

    cmd!(cmake, ["-S", ".", "-B", cmake_build_dir, "-DCMAKE_VERBOSE_MAKEFILE=ON"], env)
    cmd!(cmake, ["--build", cmake_build_dir, "--config", "Release"], env)
    cmd!(cmake, ["--install", cmake_build_dir, "--config", "Release"], env)

    {:ok, []}
  end

  defp cmd!(exec, args, env) do
    opts = [
      into: IO.stream(:stdio, :line),
      stderr_to_stdout: true,
      env: env
    ]

    case System.cmd(exec, args, opts) do
      {_, 0} -> :ok
      {_, status} -> Mix.raise("cmake failed with status #{status}")
    end
  end
end

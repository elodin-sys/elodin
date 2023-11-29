defmodule ParacosmDashboardWeb.EditorComponents do
  use Phoenix.Component

  alias Phoenix.LiveView.JS
  import ParacosmDashboardWeb.Gettext
  import ParacosmDashboardWeb.CoreComponents

  def console(assigns) do
    ~H"""
    <div class="w-full text-white overflow-auto h-[296px]">
        <div class="h-10 p-2 shadow-lg bg-orange flex items-center">
            <.icon name="hero-play-solid" class="h-6 w-6 bg-white"/>
        </div>
        <pre class="whitespace-pre-wrap overflow-auto h-64 bg-dark-matte p-2">
        error: undefined variable
        lib/paracosm_dashboard_web/components/editor.ex:16: ParacosmDashboardWeb.EditorComponents (module)


        == Compilation error in file lib/paracosm_dashboard_web/components/editor.ex ==
        ** (CompileError) lib/paracosm_dashboard_web/components/editor.ex: cannot compile module ParacosmDashboardWeb.EditorComponents (errors have been logged)
        (elixir 1.15.7) lib/kernel/parallel_compiler.ex:377: anonymous fn/5 in Kernel.ParallelCompiler.spawn_workers/8
        </pre>
    </div>
    """
  end

  def editor_wasm(assigns) do
    ~H"""
    <div phx-hook="EditorWasmHook" id="editor-container" class="bg-black h-full w-1/2">
    <canvas id="editor"/>
    </div>
    """
  end
end

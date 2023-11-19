defmodule ParacosmDashboardWeb.EditorLive do
  use ParacosmDashboardWeb, :live_view
  alias ParacosmDashboardWeb.EditorComponents
  import ParacosmDashboardWeb.CoreComponents

  def render(assigns) do
    ~H"""
    <div class="flex flex-col h-full">
        <div class="flex w-full h-full">
            <div class="w-1/2 h-full">
                <div style="height: calc(100% - 256px - 40px);" class="pt-3 bg-code">
                    <LiveMonacoEditor.code_editor
                        class="code-editor"
                        opts={
                            Map.merge(
                                LiveMonacoEditor.default_opts(),
                                %{
                                    "language" => "python",
                                    "minimap" => %{ "enabled" => false },
                                    "automaticLayout" => true,
                                }
                            )
                        }
                    />
                </div>
                <EditorComponents.console/>
            </div>
            <EditorComponents.editor_wasm/>
        </div>
    </div>
    """
  end
end

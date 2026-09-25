from module_impacted_analysis import analyze_module_impact
from state import AnalyzeArgs, AppState


def analyze_loop(app: AppState) -> None:
    assert isinstance(app.args, AnalyzeArgs)

    for file in app.files:
        analyze_module_impact(
            model=app.model,
            wav=file.wav,
            output_dir=app.output,
            max_artifacts=app.args.save_depth,
        )

from module_impacted_analysis import analyze_module_impact
from state import AnalyzeArgs, AppState


def analyze_loop(app: AppState) -> None:
    assert isinstance(app.args, AnalyzeArgs)

    for file in app.files:
        if app.args.optimized:
            from module_impacted_analysis_optimized import analyze_module_impact_optimized

            analyze_module_impact_optimized(
                model=app.model,
                wav=file.wav,
                output_dir=app.output,
                max_artifacts=app.args.save_depth,
                trials=app.args.trials,
                seed=app.args.seed,
                seconds=app.args.seconds,
            )
            continue

        analyze_module_impact(
            model=app.model,
            wav=file.wav,
            output_dir=app.output,
            max_artifacts=app.args.save_depth,
        )

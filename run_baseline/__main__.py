def main(mode: str):
    from run_baseline.run_baseline import RunBaseline

    run = RunBaseline()

    if mode == "prepare_samples":
        print("preparing samples")
        run.prepare_samples()
        return
    elif mode == "run_protect":
        print("running protect")
        run.run_protect()
        return
    elif mode == "run":
        print("running baseline")
        run.run()
        return
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline protect model.")
    parser.add_argument(
        "--mode",
        choices=["prepare_samples", "run_protect", "run"],
        default="run",
        type=str.lower,
        required=True,
        help="the mode to run with",
    )

    args = parser.parse_args()

    main(args.mode)

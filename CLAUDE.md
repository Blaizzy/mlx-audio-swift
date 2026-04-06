# Project notes for agents

- Build and test MLX targets with Xcode/xcodebuild on macOS/Apple Silicon so the default metallib is produced and bundled correctly.
- If you see `Failed to load the default metallib`, verify the bundle path (`default.metallib` / `mlx.metallib`) or use `DYLD_FRAMEWORK_PATH` for shell runs.

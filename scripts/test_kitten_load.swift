#!/usr/bin/env swift
// Quick test: load KittenTTS weights and verify no key mismatches
// Run via: swift build --target MLXAudioTTS && swift scripts/test_kitten_load.swift
// Or better: add to test suite

import Foundation
import MLX

let modelDir = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".cache/huggingface/hub/models--mlx-community--kitten-tts-nano-0.8/snapshots/f57e91b190ca3323aa94c7bbdde4565343d79588")

print("Model dir: \(modelDir.path)")
print("Config exists: \(FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("config.json").path))")
print("Weights exist: \(FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors").path))")
print("Voices exist: \(FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("voices.safetensors").path))")

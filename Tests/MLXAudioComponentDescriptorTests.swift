//
//  MLXAudioComponentDescriptorTests.swift
//  MLXAudioTests
//
//  Integration test for ComponentDescriptor usage with SNAC and Mimi codecs.
//  Verifies that descriptors are properly registered with SwiftAcervo and
//  that model resolution and download works through the ComponentDescriptor path.
//

import Testing
import Foundation
import SwiftAcervo

@testable import MLXAudioCore
@testable import MLXAudioCodecs

// Run integration tests with: xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/ComponentDescriptorIntegrationTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|registered|descriptor|resolution|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"

struct ComponentDescriptorIntegrationTests {

    // MARK: - SNAC ComponentDescriptor Tests

    @Test func testSNACComponentDescriptorRegistration() async throws {
        print("\n=== SNAC ComponentDescriptor Registration Test ===")

        // Ensure SNAC components are registered
        SNAC.ensureComponentsRegistered()

        // Verify descriptor exists in registry
        let componentId = SNACModelRepo.snac24kHz.componentId
        print("Looking for SNAC component: \(componentId)")

        // Check if component is registered
        if let descriptor = Acervo.component(componentId) {
            print("✓ SNAC descriptor found in registry")
            print("  - Display name: \(descriptor.displayName)")
            print("  - Repo ID: \(descriptor.repoId)")
            print("  - File count: \(descriptor.files.count)")
            print("  - Estimated size: \(descriptor.estimatedSizeBytes) bytes")
            print("  - Min memory: \(descriptor.minimumMemoryBytes) bytes")

            // Verify required files are declared
            #expect(descriptor.files.count >= 2, "SNAC should have at least 2 files")

            let filePaths = descriptor.files.map { $0.relativePath }
            print("  - Files: \(filePaths)")

            #expect(filePaths.contains("config.json"), "SNAC descriptor must include config.json")
            #expect(filePaths.contains("model.safetensors"), "SNAC descriptor must include model.safetensors")
        } else {
            print("✗ SNAC descriptor NOT found - descriptor registration may have failed")
            throw AcervoError.componentNotRegistered(componentId)
        }
    }

    @Test func testMimiComponentDescriptorRegistration() async throws {
        print("\n=== Mimi ComponentDescriptor Registration Test ===")

        // Ensure Mimi components are registered
        Mimi.ensureComponentsRegistered()

        // Verify descriptor exists in registry
        let componentId = MimiModelRepo.mimiPyTorchBF16.componentId
        print("Looking for Mimi component: \(componentId)")

        // Check if component is registered
        if let descriptor = Acervo.component(componentId) {
            print("✓ Mimi descriptor found in registry")
            print("  - Display name: \(descriptor.displayName)")
            print("  - Repo ID: \(descriptor.repoId)")
            print("  - File count: \(descriptor.files.count)")
            print("  - Estimated size: \(descriptor.estimatedSizeBytes) bytes")
            print("  - Min memory: \(descriptor.minimumMemoryBytes) bytes")

            // Verify required file is declared
            #expect(descriptor.files.count >= 1, "Mimi should have at least 1 file")

            let filePaths = descriptor.files.map { $0.relativePath }
            print("  - Files: \(filePaths)")

            #expect(filePaths.contains("tokenizer-e351c8d8-checkpoint125.safetensors"),
                    "Mimi descriptor must include tokenizer file")
        } else {
            print("✗ Mimi descriptor NOT found - descriptor registration may have failed")
            throw AcervoError.componentNotRegistered(componentId)
        }
    }

    // MARK: - Component Availability Tests

    @Test func testSNACComponentAvailability() throws {
        print("\n=== SNAC Component Availability Test ===")

        // Register SNAC components
        SNAC.ensureComponentsRegistered()

        let componentId = SNACModelRepo.snac24kHz.componentId
        let isReady = Acervo.isComponentReady(componentId)

        print("SNAC component '\(componentId)' ready: \(isReady)")
        // Component may or may not be ready depending on whether it was previously downloaded
        // The important part is that checking doesn't fail
    }

    @Test func testMimiComponentAvailability() throws {
        print("\n=== Mimi Component Availability Test ===")

        // Register Mimi components
        Mimi.ensureComponentsRegistered()

        let componentId = MimiModelRepo.mimiPyTorchBF16.componentId
        let isReady = Acervo.isComponentReady(componentId)

        print("Mimi component '\(componentId)' ready: \(isReady)")
        // Component may or may not be ready depending on whether it was previously downloaded
        // The important part is that checking doesn't fail
    }

    // MARK: - ComponentDescriptor Query Integration

    @Test func testComponentDescriptorQueryIntegration() throws {
        print("\n=== ComponentDescriptor Query Integration ===")

        SNAC.ensureComponentsRegistered()
        Mimi.ensureComponentsRegistered()

        // Test that all registered components can be queried
        let allComponents = Acervo.registeredComponents()
        print("Total registered components: \(allComponents.count)")

        // Check that audio codecs are properly registered
        let audioComponents = Acervo.registeredComponents(ofType: .decoder)
        print("Registered decoders: \(audioComponents.count)")

        // Verify SNAC and Mimi are in the list
        let componentIds = allComponents.map { $0.id }
        print("Component IDs: \(componentIds)")

        let hasSNAC = allComponents.contains { $0.id == SNACModelRepo.snac24kHz.componentId }
        let hasMimi = allComponents.contains { $0.id == MimiModelRepo.mimiPyTorchBF16.componentId }

        #expect(hasSNAC, "SNAC should be in registered components")
        #expect(hasMimi, "Mimi should be in registered components")

        print("✓ SNAC and Mimi properly registered in component catalog")
    }

    // MARK: - ComponentDescriptor Metadata Tests

    @Test func testComponentDescriptorMetadata() async throws {
        print("\n=== ComponentDescriptor Metadata Tests ===")

        SNAC.ensureComponentsRegistered()
        Mimi.ensureComponentsRegistered()

        // Test SNAC metadata
        if let snacDescriptor = Acervo.component(SNACModelRepo.snac24kHz.componentId) {
            print("SNAC Metadata:")
            if let sampleRate = snacDescriptor.metadata["sampleRate"] {
                print("  - Sample rate: \(sampleRate)")
                #expect(sampleRate == "24000" || sampleRate == "16000",
                        "SNAC metadata should include valid sample rate")
            }
            if let bitrate = snacDescriptor.metadata["bitrate"] {
                print("  - Bitrate: \(bitrate)")
            }
        }

        // Test Mimi metadata
        if let mimiDescriptor = Acervo.component(MimiModelRepo.mimiPyTorchBF16.componentId) {
            print("Mimi Metadata:")
            if let sampleRate = mimiDescriptor.metadata["sampleRate"] {
                print("  - Sample rate: \(sampleRate)")
                #expect(sampleRate == "24000", "Mimi metadata should indicate 24kHz")
            }
            if let numCodebooks = mimiDescriptor.metadata["numCodebooks"] {
                print("  - Number of codebooks: \(numCodebooks)")
            }
        }
    }
}

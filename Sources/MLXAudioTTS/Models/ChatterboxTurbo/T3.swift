//
//  T3.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioCore

final class T3: Module {
    let hp: T3Config
    let cfg: GPT2Config

    @ModuleInfo(key: "tfmr") private var tfmr: GPT2Model
    @ModuleInfo(key: "cond_enc") private var condEnc: T3CondEnc
    @ModuleInfo(key: "text_emb") private var textEmb: Embedding
    @ModuleInfo(key: "speech_emb") private var speechEmb: Embedding
    @ModuleInfo(key: "text_head") private var textHead: Linear
    @ModuleInfo(key: "speech_head") private var speechHead: Linear

    init(_ hp: T3Config = .turbo()) {
        self.hp = hp
        self.cfg = .medium
        self._tfmr.wrappedValue = GPT2Model(cfg)
        self._condEnc.wrappedValue = T3CondEnc(hp)
        self._textEmb.wrappedValue = Embedding(embeddingCount: hp.textTokensDictSize, dimensions: cfg.hiddenSize)
        self._speechEmb.wrappedValue = Embedding(embeddingCount: hp.speechTokensDictSize, dimensions: cfg.hiddenSize)
        self._textHead.wrappedValue = Linear(cfg.hiddenSize, hp.textTokensDictSize, bias: false)
        self._speechHead.wrappedValue = Linear(cfg.hiddenSize, hp.speechTokensDictSize, bias: true)
    }

    private func prepareConditioning(_ cond: T3Cond) -> MLXArray {
        var updatedCond = cond
        if let tokens = updatedCond.condPromptSpeechTokens, updatedCond.condPromptSpeechEmb == nil {
            updatedCond.condPromptSpeechEmb = speechEmb(tokens)
        }
        return condEnc(updatedCond)
    }

    private func prepareInputEmbeds(
        cond: T3Cond,
        textTokens: MLXArray,
        speechTokens: MLXArray
    ) -> (MLXArray, Int) {
        var condEmb = prepareConditioning(cond)
        let textEmbeds = textEmb(textTokens)
        let speechEmbeds = speechEmb(speechTokens)

        let condLen = condEmb.dim(1)

        if condEmb.dim(0) != textEmbeds.dim(0) {
            condEmb = condEmb.broadcasted(to: [textEmbeds.dim(0), condEmb.dim(1), condEmb.dim(2)])
        }

        let embeds = MLX.concatenated([condEmb, textEmbeds, speechEmbeds], axis: 1)
        return (embeds, condLen)
    }

    func inferenceTurboStream(
        cond: T3Cond,
        textTokens: MLXArray,
        temperature: Float = 0.8,
        topK: Int = 1000,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.2,
        maxGenLen: Int = 1000,
        chunkSize: Int = 40
    ) -> AsyncStream<(MLXArray, Bool)> {
        AsyncStream { continuation in
            Task {
                var textTokens = textTokens
                if textTokens.ndim == 1 {
                    textTokens = textTokens.expandedDimensions(axis: 0)
                }

                let batch = textTokens.dim(0)
                let speechStart = MLXArray.ones([batch, 1]).asType(.int32) * MLXArray(Int32(hp.startSpeechToken))

                let (embeds, _) = prepareInputEmbeds(cond: cond, textTokens: textTokens, speechTokens: speechStart)
                let (hiddenStates, cache) = tfmr(inputsEmbeds: embeds, cache: nil)

                let speechHidden = hiddenStates[0..., (hiddenStates.dim(1) - 1)..<hiddenStates.dim(1), 0...]
                let logits = speechHead(speechHidden)

                var generatedTokens: [Int] = []
                let nextToken = sampleToken(
                    logits: logits[0],
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    generatedTokens: nil,
                    repetitionPenalty: repetitionPenalty
                )

                generatedTokens.append(nextToken)

                var currentToken = MLXArray(Int32(nextToken)).reshaped([1, 1])
                var activeCache = cache

                var chunkBuffer: [MLXArray] = [currentToken]

                for _ in 0..<maxGenLen {
                    let speechEmbed = speechEmb(currentToken)
                    let (nextHidden, updatedCache) = tfmr(inputsEmbeds: speechEmbed, cache: activeCache)
                    activeCache = updatedCache

                    let nextLogits = speechHead(nextHidden)
                    let sampled = sampleToken(
                        logits: nextLogits[0],
                        temperature: temperature,
                        topK: topK,
                        topP: topP,
                        generatedTokens: generatedTokens,
                        repetitionPenalty: repetitionPenalty
                    )

                    generatedTokens.append(sampled)

                    currentToken = MLXArray(Int32(sampled)).reshaped([1, 1])
                    chunkBuffer.append(currentToken)

                    if sampled == hp.stopSpeechToken {
                        if chunkBuffer.count > 1 {
                            let chunk = MLX.concatenated(chunkBuffer.dropLast(), axis: 1)
                            eval(chunk)
                            continuation.yield((chunk, true))
                        }
                        continuation.finish()
                        return
                    }

                    if chunkBuffer.count >= chunkSize {
                        let chunk = MLX.concatenated(chunkBuffer, axis: 1)
                        eval(chunk)
                        continuation.yield((chunk, false))
                        chunkBuffer.removeAll(keepingCapacity: true)
                    }
                }

                if !chunkBuffer.isEmpty {
                    let chunk = MLX.concatenated(chunkBuffer, axis: 1)
                    eval(chunk)
                    continuation.yield((chunk, true))
                }

                continuation.finish()
            }
        }
    }

    func inferenceTurbo(
        cond: T3Cond,
        textTokens: MLXArray,
        temperature: Float = 0.8,
        topK: Int = 1000,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.2,
        maxGenLen: Int = 1000
    ) -> MLXArray {
        var textTokens = textTokens
        if textTokens.ndim == 1 {
            textTokens = textTokens.expandedDimensions(axis: 0)
        }

        let batch = textTokens.dim(0)
        let speechStart = MLXArray.ones([batch, 1]).asType(.int32) * MLXArray(Int32(hp.startSpeechToken))

        let (embeds, _) = prepareInputEmbeds(cond: cond, textTokens: textTokens, speechTokens: speechStart)
        let (hiddenStates, cache) = tfmr(inputsEmbeds: embeds, cache: nil)

        let speechHidden = hiddenStates[0..., (hiddenStates.dim(1) - 1)..<hiddenStates.dim(1), 0...]
        let logits = speechHead(speechHidden)

        var generatedTokens: [Int] = []
        var nextToken = sampleToken(
            logits: logits[0],
            temperature: temperature,
            topK: topK,
            topP: topP,
            generatedTokens: nil,
            repetitionPenalty: repetitionPenalty
        )

        generatedTokens.append(nextToken)
        var currentToken = MLXArray(Int32(nextToken)).reshaped([1, 1])
        var activeCache = cache

        for _ in 0..<maxGenLen {
            let speechEmbed = speechEmb(currentToken)
            let (nextHidden, updatedCache) = tfmr(inputsEmbeds: speechEmbed, cache: activeCache)
            activeCache = updatedCache

            let nextLogits = speechHead(nextHidden)
            nextToken = sampleToken(
                logits: nextLogits[0],
                temperature: temperature,
                topK: topK,
                topP: topP,
                generatedTokens: generatedTokens,
                repetitionPenalty: repetitionPenalty
            )

            generatedTokens.append(nextToken)
            currentToken = MLXArray(Int32(nextToken)).reshaped([1, 1])

            if nextToken == hp.stopSpeechToken {
                break
            }
        }

        if generatedTokens.last == hp.stopSpeechToken {
            generatedTokens.removeLast()
        }

        let tokens = generatedTokens.map { Int32($0) }
        return MLXArray(tokens).reshaped([1, tokens.count])
    }

    private func sampleToken(
        logits: MLXArray,
        temperature: Float,
        topK: Int,
        topP: Float,
        generatedTokens: [Int]?,
        repetitionPenalty: Float
    ) -> Int {
        var filteredLogits = logits

        if let generatedTokens, repetitionPenalty != 1.0, !generatedTokens.isEmpty {
            let vocabSize = filteredLogits.dim(-1)
            var tokenMask = [Float](repeating: 0, count: vocabSize)
            for token in Set(generatedTokens) where token >= 0 && token < vocabSize {
                tokenMask[token] = 1.0
            }

            let mask = MLXArray(tokenMask).reshaped([1, vocabSize])
            let penalty = MLXArray(repetitionPenalty)
            let penalized = MLX.where(filteredLogits .< 0, filteredLogits * penalty, filteredLogits / penalty)
            filteredLogits = MLX.where(mask .> 0, penalized, filteredLogits)
        }

        if temperature > 0 && temperature != 1.0 {
            filteredLogits = filteredLogits / MLXArray(temperature)
        }

        if topK > 0 {
            let vocabSize = filteredLogits.dim(-1)
            let k = min(topK, vocabSize)
            if k > 0, k < vocabSize {
                let partitioned = argPartition(filteredLogits, kth: -k, axis: -1)
                let kthPosition = vocabSize - k
                let kthIndices = partitioned[0..., kthPosition..<(kthPosition + 1)]
                let kthValues = takeAlong(filteredLogits, kthIndices, axis: -1)
                let keepMask = filteredLogits .>= kthValues
                filteredLogits = MLX.where(keepMask, filteredLogits, MLXArray(-Float.infinity))
            }
        }

        if topP < 1.0 {
            let sortedIndices = argSort(-filteredLogits, axis: -1)
            let sortedLogits = takeAlong(filteredLogits, sortedIndices, axis: -1)
            let sortedProbs = softmax(sortedLogits, axis: -1)
            let cumulativeProbs = cumsum(sortedProbs, axis: -1)

            var sortedIndicesToRemove = cumulativeProbs .> MLXArray(topP)
            let batch = filteredLogits.dim(0)
            let vocabSize = filteredLogits.dim(-1)

            if vocabSize > 1 {
                let first = MLXArray.zeros([batch, 1], type: Bool.self)
                let shifted = sortedIndicesToRemove[0..., 0..<(vocabSize - 1)]
                sortedIndicesToRemove = MLX.concatenated([first, shifted], axis: -1)
            }

            let filteredSorted = MLX.where(sortedIndicesToRemove, MLXArray(-Float.infinity), sortedLogits)
            let inverseIndices = argSort(sortedIndices, axis: -1)
            filteredLogits = takeAlong(filteredSorted, inverseIndices, axis: -1)
        }

        let next = categorical(filteredLogits)
        eval(next)
        return Int(next.item(Int32.self))
    }
}

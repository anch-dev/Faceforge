import Foundation

/// Rolling frame statistics: average FPS, 1%-low FPS, latency percentiles.
/// Same definitions as faceforge_eval.pipeline.summarize.
public struct FrameStats {
    public private(set) var frameMs: [Double] = []
    public let window: Int

    public init(window: Int = 600) { self.window = window }

    public mutating func add(ms: Double) {
        frameMs.append(ms)
        if frameMs.count > window { frameMs.removeFirst(frameMs.count - window) }
    }

    public mutating func reset() { frameMs.removeAll() }

    public var avgFPS: Double {
        let total = frameMs.reduce(0, +)
        return total > 0 ? 1000.0 * Double(frameMs.count) / total : 0
    }

    /// FPS implied by the mean of the slowest 1% of frames (at least one).
    public var onePercentLowFPS: Double {
        guard !frameMs.isEmpty else { return 0 }
        let sorted = frameMs.sorted()
        let n = max(1, Int((Double(sorted.count) * 0.01).rounded(.up)))
        let slow = sorted.suffix(n)
        return 1000.0 / (slow.reduce(0, +) / Double(n))
    }

    /// Linear-interpolated percentile (numpy default).
    public func percentile(_ p: Double) -> Double {
        guard !frameMs.isEmpty else { return 0 }
        let s = frameMs.sorted()
        let rank = p / 100 * Double(s.count - 1)
        let lo = Int(rank.rounded(.down)), hi = Int(rank.rounded(.up))
        return s[lo] + (s[hi] - s[lo]) * (rank - Double(lo))
    }
}

/// How hard to drive the pipeline at each thermal state.
/// Raw values mirror ProcessInfo.ThermalState (nominal 0 ... critical 3).
public struct ThermalPolicy: Equatable {
    public let maxFPS: Int
    public let detectEvery: Int
    public let paused: Bool

    public static func forState(_ rawState: Int, userDetectEvery: Int) -> ThermalPolicy {
        switch rawState {
        case 0: return ThermalPolicy(maxFPS: 30, detectEvery: max(1, userDetectEvery), paused: false)
        case 1: return ThermalPolicy(maxFPS: 24, detectEvery: max(2, userDetectEvery), paused: false)
        case 2: return ThermalPolicy(maxFPS: 15, detectEvery: max(4, userDetectEvery), paused: false)
        default: return ThermalPolicy(maxFPS: 0, detectEvery: max(8, userDetectEvery), paused: true)
        }
    }
}

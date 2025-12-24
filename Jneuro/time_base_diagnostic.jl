# TIME BASE DIAGNOSTIC SCRIPT
# Run this to understand what's happening with your time coordinates

println("="^70)
println("TIME BASE DIAGNOSTIC")
println("="^70)

# Simulate your setup
fs = 500.0
original_length = 2000
filtered_length = 1700

println("\n1. ORIGINAL RECORDING")
println("   Samples: $original_length")
println("   Duration: $(original_length/fs)s at $(fs)Hz")
println("   Time range: -2.0s to +2.0s (event-aligned)")

println("\n2. AFTER FILTERING (trim 150 samples each end)")
println("   Samples: $filtered_length")
println("   Duration: $(filtered_length/fs)s at $(fs)Hz") 
println("   Time range: -1.7s to +1.7s (trimmed)")

println("\n3. SWR DETECTION INTERNAL TIME AXIS")
println("   ALWAYS starts at 0!")
println("   time = (0:length(signal)-1) / fs")
println("   For $filtered_length samples: [0.0, $(filtered_length/fs)]")

println("\n4. THE PROBLEM")
println("   Behavioral events at: -1.5s, -1.0s, 0.5s, ...")
println("   SWR detection expects: 0.0s, 0.2s, 2.2s, ...")
println("   ❌ MISMATCH!")

println("\n5. THE SOLUTION")
println("   Shift behavioral events:")
println("   event_adjusted = event_original - time_bins[1]")
println("   Example: -1.5 - (-1.7) = 0.2 ✓")

println("\n6. VERIFICATION CHECKLIST")
println("   After running neural_analysis_complete.jl, check:")
println("   ✓ Do shifted events show positive times? (0 to 3.4)")
println("   ✓ Does verification pass before analysis?")  
println("   ✓ Are there NO 'outside bounds' warnings?")
println("   ✓ Do you actually detect SWRs near events?")

println("\n" * "="^70)
println("KEY INSIGHT")
println("="^70)
println("""
The SWR detection function CANNOT know your original time base!
It receives a signal array and creates its own time axis from 0.

YOU must shift your behavioral event times to match this 0-based axis.

Original time:    [-1.7 ━━━━━━━━━━━━━ +1.7] (3.4s duration)
SWR internal:     [0.0 ━━━━━━━━━━━━━ 3.4]  (same duration, different origin)
                   ^                    ^
                   |                    |
                   These must align!
                   
Event at -1.5s → shift by 1.7 → becomes 0.2s ✓
""")

println("="^70)

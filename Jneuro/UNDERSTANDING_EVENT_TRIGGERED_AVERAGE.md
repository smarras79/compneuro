# Understanding Event-Triggered Average vs Firing Rate Plot

## 🤔 **Your Confusion is Understandable!**

These two plots show **different aspects** of the same data. Let me explain clearly.

---

## 📊 **The Two Plots Explained**

### **Plot 1: Firing Rate Over Time**
**What it shows:** The ENTIRE neural signal from -2s to +2s

```
Firing Rate (Hz)
    ↑
 50 |     ╱╲    |    ╱╲         |
 40 |    ╱  ╲   |   ╱  ╲        |
 30 |   ╱    ╲  |  ╱    ╲       |
 20 |__╱______╲_|_╱______╲______|____
    -2  -1   0  +1  +2 seconds
         ↑         ↑         ↑
       Event 1   Event 2   Event 3
       (red lines show when motion occurred)
```

**Purpose:** See the ENTIRE recording and WHERE events occurred

---

### **Plot 2: Event-Triggered Average (ETA)**
**What it shows:** A SHORT WINDOW (±0.5s) around EACH event, then AVERAGED

```
Step 1: Extract windows around each event
Event 1:     Event 2:     Event 3:
  ╱╲           ╱╲           ╱╲
 ╱  ╲         ╱  ╲         ╱  ╲
╱____╲       ╱____╲       ╱____╲
-0.5 0 +0.5  -0.5 0 +0.5  -0.5 0 +0.5
  ↑            ↑            ↑
Event       Event        Event

Step 2: Average them together
     ╱╲
    ╱  ╲
   ╱    ╲
  ╱______╲
-0.5  0  +0.5
      ↑
  Event time
```

**Purpose:** See if there's a CONSISTENT PATTERN around events

---

## 🎯 **Key Differences**

| Aspect | Firing Rate Plot | Event-Triggered Average |
|--------|------------------|------------------------|
| **Time range** | Full recording (-2 to +2s) | Short window (±0.5s) |
| **What it shows** | Complete signal | Zoomed around events |
| **Events** | Multiple (scattered) | All aligned to t=0 |
| **Data** | Raw signal | Averaged signal |
| **Purpose** | Overview | Statistical pattern |
| **Question answered** | "When did events occur?" | "What happens around events?" |

---

## 🔬 **Scientific Purpose**

### **Why Short Window (-0.5 to +0.5s)?**

**The ETA answers:** "Is there a CONSISTENT neural response RIGHT AROUND when motion occurs?"

**Example interpretations:**

#### **Scenario A: Peak at t=0**
```
Firing Rate (Hz)
    ↑
 50 |        ╱╲
 40 |       ╱  ╲
 30 |      ╱    ╲
 20 |_____╱______╲_____
   -0.5   0    +0.5
          ↑
      Motion onset
```
**Interpretation:** Neural activity PEAKS when motion happens
**Meaning:** Neurons are ACTIVE during motion

---

#### **Scenario B: Peak BEFORE t=0**
```
Firing Rate (Hz)
    ↑
 50 |    ╱╲
 40 |   ╱  ╲
 30 |  ╱    ╲___
 20 |_╱         ╲____
   -0.5   0    +0.5
          ↑
      Motion onset
```
**Interpretation:** Activity peaks BEFORE motion
**Meaning:** Neurons show **PREPARATORY** activity (planning)

---

#### **Scenario C: Peak AFTER t=0**
```
Firing Rate (Hz)
    ↑
 50 |            ╱╲
 40 |           ╱  ╲
 30 |      ____╱    ╲
 20 |_____╱__________╲
   -0.5   0    +0.5
          ↑
      Motion onset
```
**Interpretation:** Activity peaks AFTER motion
**Meaning:** Neurons respond to **FEEDBACK** from motion

---

#### **Scenario D: No Peak (Flat)**
```
Firing Rate (Hz)
    ↑
 30 |___________________
 20 |___________________
 10 |___________________
    -0.5   0    +0.5
          ↑
      Motion onset
```
**Interpretation:** No consistent pattern
**Meaning:** Neural activity is **UNRELATED** to motion

---

## 💡 **Concrete Example**

### **Your Data: 399 Motion Events**

**Firing Rate Plot shows:**
```
Time:    -2s     -1s      0s      +1s     +2s
Signal:  ~~~~~~|~~~~|~~~~~~~~~|~~~~~~~|~~~~~
Events:     ↑       ↑    ↑         ↑
         (399 events scattered across time)
```

**Event-Triggered Average does:**

1. **Find all 399 motion events**
2. **Extract ±0.5s around EACH one:**
   - Event 1: signal from -0.15s to +0.85s → shift to (-0.5, +0.5)
   - Event 2: signal from +0.23s to +1.23s → shift to (-0.5, +0.5)
   - Event 3: signal from -0.87s to +0.13s → shift to (-0.5, +0.5)
   - ... all 399 events ...

3. **Average them together:**
   - Now all events are aligned at t=0
   - We see the TYPICAL pattern

4. **Plot the average:**
   - Time axis: -0.5 to +0.5 (relative to event)
   - Signal: average firing rate
   - Shaded area: variability (SEM)

---

## 🎨 **Visual Analogy**

Think of it like analyzing a **baseball swing**:

### **Firing Rate Plot = Video of Entire Game**
- Shows all 9 innings
- You see when players swung
- Hard to see swing details

### **Event-Triggered Average = Slow-Motion Replay**
- Zooms to ±0.5 seconds around EACH swing
- Averages 399 swings together
- Shows the TYPICAL swing pattern

---

## 🔍 **Why Not Show Full -2 to +2s Range?**

**Short answer:** You'd see nothing useful!

**Long answer:**

If we showed -2 to +2 seconds:
1. Events are scattered across this range
2. When aligned to t=0, data from other events would be far away
3. Most of the plot would be empty or meaningless
4. The interesting dynamics happen within ±0.5s

**The ±0.5s window focuses on the ACTION:**
- Before: preparation/anticipation
- During: execution
- After: response/feedback

---

## ⚙️ **Customizing the Window**

If you want to see a WIDER window:

In `enhanced_visualization.jl`, line ~250:
```julia
function plot_event_triggered_average(...;
                                     window=(-0.5, 0.5),  # ← Change this!
                                     ...)
```

**Options:**
```julia
window=(-1.0, 1.0)    # ±1 second (wider)
window=(-0.25, 0.25)  # ±250ms (tighter)
window=(-0.5, 1.0)    # Asymmetric (more after)
window=(-2.0, 2.0)    # Full range (if you really want)
```

**Typical ranges in neuroscience:**
- Fast processes: ±100-250ms
- Motor responses: ±500ms (current)
- Cognitive processes: ±1-2s

---

## 📊 **What Each Plot Tells You**

### **Firing Rate Plot**
**Answers:**
- ✅ How does activity change over the entire trial?
- ✅ When do events occur?
- ✅ How frequent are events?
- ✅ Overall temporal structure

**Doesn't answer:**
- ❌ Is there a consistent pattern around events?
- ❌ What's the typical response?

### **Event-Triggered Average**
**Answers:**
- ✅ Is there a consistent neural response to events?
- ✅ What's the timing (before/during/after)?
- ✅ How variable is the response? (SEM)
- ✅ Statistical relationship

**Doesn't answer:**
- ❌ When do events occur in the trial?
- ❌ Individual event responses

---

## 🎓 **Scientific Interpretation Guide**

### **What to Look For in ETA:**

1. **Peak location:**
   - Before t=0: Preparatory activity
   - At t=0: Direct response
   - After t=0: Feedback/consequence

2. **Peak amplitude:**
   - High: Strong relationship
   - Low: Weak relationship
   - Flat: No relationship

3. **Width:**
   - Narrow: Precise timing
   - Wide: Prolonged response

4. **Symmetry:**
   - Symmetric: Activity brackets event
   - Asymmetric: One-sided relationship

5. **SEM (shaded area):**
   - Narrow: Consistent across events
   - Wide: Variable responses

---

## 📈 **Example Interpretations**

### **Example 1: Motor Neuron**
```
     ╱╲
    ╱  ╲
   ╱    ╲
  ╱______╲
-0.5 0  +0.5
     ↑
```
**Interpretation:** Motor neuron fires DURING motion
**Timing:** Peak at t=0
**Function:** Drives the movement

### **Example 2: Planning Neuron**
```
  ╱╲
 ╱  ╲___
╱       ╲
-0.5 0  +0.5
     ↑
```
**Interpretation:** Planning neuron fires BEFORE motion
**Timing:** Peak at t=-200ms
**Function:** Prepares for movement

### **Example 3: Sensory Feedback**
```
        ╱╲
   ____╱  ╲
  ╱        ╲
-0.5 0  +0.5
     ↑
```
**Interpretation:** Sensory neuron fires AFTER motion
**Timing:** Peak at t=+200ms
**Function:** Processes movement feedback

---

## 🔬 **Your Specific Data**

Based on your stimulus-locked design:

**Time structure:**
```
-2s     -1s      0s      +1s     +2s
|-------|-------|-------|-------|
        Stimulus appears at t=0
        Motion events: -0.15s to +1.49s
```

**What ETA shows:**
- Windows around each of 399 motion events
- Averaged to see typical pattern
- Focused on ±0.5s for clarity

**To interpret YOUR ETA:**
1. Look at peak location
2. Is it before, during, or after t=0?
3. This tells you the neural-behavioral timing

---

## 💡 **Summary**

### **Firing Rate Plot:**
```
"Here's ALL the neural activity and when motion happened"
- Full timeline: -2 to +2s
- Shows: Complete recording
- Purpose: Overview
```

### **Event-Triggered Average:**
```
"Here's what typically happens RIGHT AROUND motion"
- Short window: ±0.5s
- Shows: Average pattern
- Purpose: Statistical relationship
```

### **They're Complementary:**
- Firing rate: **WHERE** events are in time
- ETA: **WHAT** happens around events

### **The ±0.5s Window:**
- Focuses on the interesting dynamics
- Standard in neuroscience
- Customizable if needed

---

## ✅ **Quick Check**

**You understand correctly if you can answer:**

Q: Why is ETA -0.5 to +0.5 instead of -2 to +2?
A: To focus on what happens RIGHT AROUND events, not the entire trial

Q: What does a peak at t=0 mean?
A: Neural activity is highest when motion occurs

Q: What does a peak at t=-0.3s mean?
A: Neural activity precedes motion (preparatory)

Q: Why average multiple events?
A: To see if there's a CONSISTENT pattern, not just noise

---

## 🎯 **Bottom Line**

**Firing Rate Plot:** "When did things happen?"  
**Event-Triggered Average:** "What typically happens when they happen?"

Both are essential for understanding neural-behavioral relationships!

---

**Still confused? The key insight:**
- ETA is like a "zoom and average" operation
- It answers: "Is there a consistent neural signature around behavior?"
- The ±0.5s window is where the action is!

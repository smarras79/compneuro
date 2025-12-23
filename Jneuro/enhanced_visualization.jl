"""
Enhanced Visualization Module with Behavioral Event Markers
PUBLICATION QUALITY VERSION

Settings optimized for journal publication:
- DPI: 600 (high resolution)
- Margins: 20mm left (room for labels), 12mm others
- Fonts: 16pt titles, 14pt labels, 12pt ticks
- Line widths: 2.5-3pt
- Colors: High contrast, colorblind-friendly

Author: Enhanced Neural Analysis Toolkit
"""

using Plots
using Statistics
using Printf

# Set global publication defaults
default(
    fontfamily="Computer Modern",
    linewidth=2.5,
    framestyle=:box,
    grid=true,
    gridstyle=:dot,
    gridalpha=0.25,
    gridlinewidth=0.5,
    legend_font_pointsize=11,
    legend_foreground_color=:transparent
)

"""
    plot_signal_with_events(time, signal, behavioral_events; kwargs...)

Plot neural signal with vertical lines marking behavioral events.
Publication-quality settings.
"""
function plot_signal_with_events(time, signal, behavioral_events; 
                                title::String="Neural Signal with Behavioral Events",
                                ylabel::String="Firing Rate (Hz)",
                                time_range=nothing,
                                swr_events=nothing,
                                figsize::Tuple=(2000, 900))
    
    # Apply time range if specified
    if !isnothing(time_range)
        mask = (time .>= time_range[1]) .& (time .<= time_range[2])
        time = time[mask]
        signal = signal[mask]
    end
    
    # Create main plot with publication settings
    p = plot(time, signal, 
             label="Neural Signal",
             linewidth=2.5,
             color=:black,
             xlabel="Time (s)",
             ylabel=ylabel,
             title=title,
             legend=:topright,
             size=figsize,
             dpi=600,
             left_margin=20Plots.mm,
             right_margin=12Plots.mm,
             top_margin=12Plots.mm,
             bottom_margin=12Plots.mm,
             titlefontsize=16,
             guidefontsize=14,
             tickfontsize=12,
             legendfontsize=11,
             framestyle=:box,
             grid=true,
             gridstyle=:dot,
             gridalpha=0.25,
             minorgrid=false)
             minorgrid=false)
    
    # Add behavioral event markers
    if !isnothing(behavioral_events)
        colors = Dict(
            "motion_onset" => RGB(0.9, 0.1, 0.1),    # Bright red
            "motion_offset" => RGB(0.1, 0.6, 0.9),   # Bright cyan  
            "trial_onset" => RGB(0.1, 0.7, 0.1),     # Green
            "event_onset" => RGB(1.0, 0.5, 0.0)      # Orange
        )
        
        labels = Dict(
            "motion_onset" => "Motion Onset",
            "motion_offset" => "Motion Offset",
            "trial_onset" => "Trial Onset",
            "event_onset" => "Event Onset"
        )
        
        first_of_type = Dict()
        
        for (event_type, event_times) in behavioral_events
            # Skip metadata
            if event_type == "n_events" || !isa(event_times, AbstractVector)
                continue
            end
            
            color = get(colors, event_type, RGB(0.5, 0.0, 0.5))
            label = get(labels, event_type, event_type)
            
            # Filter to visible time range
            if !isnothing(time_range)
                event_times = event_times[(event_times .>= time_range[1]) .& 
                                         (event_times .<= time_range[2])]
            end
            
            if length(event_times) == 0
                continue
            end
            
            # Plot vertical lines for each event
            for (idx, t) in enumerate(event_times)
                # Only add to legend for first event of this type
                show_label = !haskey(first_of_type, event_type)
                first_of_type[event_type] = true
                
                vline!([t], 
                      color=color,
                      alpha=0.7,
                      linewidth=2.5,
                      linestyle=:dash,
                      label=show_label ? label : "")
            end
        end
    end
    
    # Add SWR markers if provided
    if !isnothing(swr_events) && haskey(swr_events, "events")
        fs = 1000.0  # Assume 1kHz if not specified
        
        for (idx, event) in enumerate(swr_events["events"])
            event_time = event.peak_sample / fs
            
            # Check if in time range
            if !isnothing(time_range) && 
               (event_time < time_range[1] || event_time > time_range[2])
                continue
            end
            
            # Mark with vertical line
            vline!([event_time],
                  color=RGB(0.8, 0.0, 0.8),  # Magenta
                  alpha=0.5,
                  linewidth=2,
                  label=(idx == 1 ? "SWR" : ""))
        end
    end
    
    return p
end

"""
    plot_raster_with_events(time, signal, behavioral_events, swr_events; 
                            window_size=0.5)

Create a raster plot showing signal segments around behavioral events with SWRs marked.
"""
function plot_raster_with_events(time, signal, behavioral_events, swr_events;
                                 window_size=0.5,
                                 fs=1000.0,
                                 figsize=(1200, 800))
    
    # Collect all event times and types
    all_events = []
    for (event_type, event_times) in behavioral_events
        if event_type == "n_events" || !isa(event_times, AbstractVector)
            continue
        end
        
        for t in event_times
            push!(all_events, (time=t, type=event_type))
        end
    end
    
    if length(all_events) == 0
        return plot(title="No behavioral events to display")
    end
    
    # Sort by time
    sort!(all_events, by=x->x.time)
    
    # Take first 20 events max for visibility
    n_display = min(20, length(all_events))
    all_events = all_events[1:n_display]
    
    # Create subplots - one per event
    plots = []
    
    for (idx, event) in enumerate(all_events)
        # Get window around this event
        t_start = event.time - window_size/2
        t_end = event.time + window_size/2
        
        mask = (time .>= t_start) .& (time .<= t_end)
        if sum(mask) == 0
            continue
        end
        
        t_window = time[mask]
        s_window = signal[mask]
        
        # Center time on event
        t_centered = t_window .- event.time
        
        # Create subplot
        p = plot(t_centered, s_window,
                linewidth=1.5,
                color=:black,
                xlabel=(idx == n_display ? "Time rel. to event (s)" : ""),
                ylabel="Amp",
                title="Event $idx: $(event.type) at $(round(event.time, digits=2))s",
                legend=false,
                size=(1200, 100),
                titlefontsize=8,
                grid=true)
        
        # Mark event center
        vline!([0.0], color=:red, linewidth=2, alpha=0.7)
        
        # Mark SWRs in this window
        if !isnothing(swr_events) && haskey(swr_events, "events")
            for swr in swr_events["events"]
                swr_time = swr.peak_sample / fs
                if swr_time >= t_start && swr_time <= t_end
                    t_rel = swr_time - event.time
                    vline!([t_rel], color=:magenta, linewidth=1.5, alpha=0.5)
                end
            end
        end
        
        push!(plots, p)
    end
    
    # Combine into single figure
    return plot(plots..., layout=(n_display, 1), size=figsize)
end

"""
    plot_spectrogram_with_events(times, frequencies, spectrogram, behavioral_events;
                                title="Spectrogram with Behavioral Events")

Plot spectrogram with vertical lines marking behavioral events.
Publication quality.
"""
function plot_spectrogram_with_events(times, frequencies, spectrogram, 
                                     behavioral_events=nothing;
                                     title="Spectrogram with Behavioral Events",
                                     freq_range=(0, 300),
                                     figsize=(2000, 900))
    
    # Filter frequency range
    freq_mask = (frequencies .>= freq_range[1]) .& (frequencies .<= freq_range[2])
    freq_display = frequencies[freq_mask]
    spec_display = spectrogram[freq_mask, :]
    
    # Create spectrogram plot with publication settings
    p = heatmap(times, freq_display, log10.(spec_display .+ 1e-10),
               xlabel="Time (s)",
               ylabel="Frequency (Hz)",
               title=title,
               colorbar_title="log₁₀(Power)",
               c=:viridis,  # Colorblind-friendly colormap
               size=figsize,
               dpi=600,
               left_margin=20Plots.mm,
               right_margin=18Plots.mm,  # Extra room for colorbar
               top_margin=12Plots.mm,
               bottom_margin=12Plots.mm,
               titlefontsize=16,
               guidefontsize=14,
               tickfontsize=12,
               colorbar_titlefontsize=12,
               framestyle=:box)
    
    # Add behavioral event markers
    if !isnothing(behavioral_events)
        colors = Dict(
            "motion_onset" => RGB(1.0, 0.2, 0.2),      # Bright red
            "motion_offset" => RGB(0.2, 0.8, 1.0),     # Bright cyan
            "trial_onset" => RGB(0.2, 1.0, 0.2),       # Bright green
            "event_onset" => RGB(1.0, 0.6, 0.0)        # Orange
        )
        
        for (event_type, event_times) in behavioral_events
            if event_type == "n_events" || !isa(event_times, AbstractVector)
                continue
            end
            
            color = get(colors, event_type, RGB(1.0, 1.0, 1.0))
            
            # Filter to visible time range
            event_times = event_times[(event_times .>= times[1]) .& 
                                     (event_times .<= times[end])]
            
            for t in event_times
                vline!([t], 
                      color=color, 
                      alpha=0.8, 
                      linewidth=2.5, 
                      linestyle=:dash)
            end
        end
    end
    
    return p
end

"""
    create_summary_figure(time, signal, behavioral_events, swr_results;
                         time_range=(0, 10))

Create comprehensive summary figure with:
1. Signal with behavioral events and SWRs marked
2. Zoomed view around first behavioral event
3. Event-triggered average

Publication quality - suitable for papers.
"""
function create_summary_figure(time, signal, behavioral_events, swr_results;
                              time_range=(0, 10),
                              fs=1000.0,
                              figsize=(2000, 1500))
    
    # Panel 1: Full signal with events
    p1 = plot_signal_with_events(time, signal, behavioral_events;
                                 title="A. Neural Signal with Behavioral Events",
                                 time_range=time_range,
                                 swr_events=swr_results,
                                 figsize=(2000, 500))
    
    # Panel 2: Zoomed view around first event
    first_event_time = nothing
    for (event_type, event_times) in behavioral_events
        if event_type != "n_events" && isa(event_times, AbstractVector) && 
           length(event_times) > 0
            first_event_time = event_times[1]
            break
        end
    end
    
    if !isnothing(first_event_time)
        zoom_range = (first_event_time - 0.5, first_event_time + 0.5)
        p2 = plot_signal_with_events(time, signal, behavioral_events;
                                     title="B. Zoomed View Around First Event",
                                     time_range=zoom_range,
                                     swr_events=swr_results,
                                     figsize=(2000, 500))
    else
        p2 = plot(title="B. No events to zoom", 
                 size=(2000, 500),
                 left_margin=20Plots.mm,
                 titlefontsize=16,
                 dpi=600,
                 framestyle=:box)
    end
    
    # Panel 3: Event-triggered average
    p3 = plot_event_triggered_average(time, signal, behavioral_events, swr_results;
                                      fs=fs,
                                      figsize=(2000, 500))
    
    # Update title for panel C
    plot!(p3, title="C. Event-Triggered Average (n=$(length(p3.series_list[1][:y])) events)")
    
    # Combine all panels with proper spacing
    return plot(p1, p2, p3, 
               layout=(3, 1), 
               size=figsize,
               dpi=600,
               left_margin=5Plots.mm,
               right_margin=5Plots.mm,
               top_margin=5Plots.mm,
               bottom_margin=5Plots.mm)
end

"""
    create_statistics_text(behavioral_events, swr_results)

Create formatted text for statistics panel.
"""
function create_statistics_text(behavioral_events, swr_results)
    # Count events
    n_onset = 0
    n_offset = 0
    n_other = 0
    
    for (event_type, event_times) in behavioral_events
        if event_type == "n_events" || !isa(event_times, AbstractVector)
            continue
        end
        
        if occursin("onset", event_type)
            n_onset += length(event_times)
        elseif occursin("offset", event_type)
            n_offset += length(event_times)
        else
            n_other += length(event_times)
        end
    end
    
    # Count SWRs
    n_swr = 0
    if !isnothing(swr_results) && haskey(swr_results, "events")
        n_swr = length(swr_results["events"])
    end
    
    # Create text
    stats_text = """
    SUMMARY STATISTICS
    
    Behavioral Events:
      • Motion Onset: $n_onset events
      • Motion Offset: $n_offset events
      • Other Events: $n_other events
      • Total: $(n_onset + n_offset + n_other) events
    
    Neural Events:
      • Sharp-Wave Ripples: $n_swr detected
    """
    
    return stats_text
end

"""
    plot_event_triggered_average(time, signal, behavioral_events, swr_results;
                                window=[-0.5, 0.5])

Create event-triggered average of neural signal with publication quality.
"""
function plot_event_triggered_average(time, signal, behavioral_events, swr_results;
                                     window=(-0.5, 0.5),
                                     fs=1000.0,
                                     figsize=(2000, 800))
    
    # Collect all motion onset events
    event_times = Float64[]
    for (event_type, times) in behavioral_events
        if occursin("onset", event_type) && isa(times, AbstractVector)
            append!(event_times, times)
        end
    end
    
    if length(event_times) == 0
        return plot(title="No onset events found", 
                   size=figsize,
                   left_margin=20Plots.mm,
                   bottom_margin=12Plots.mm,
                   titlefontsize=16,
                   dpi=600)
    end
    
    # Extract windows around events
    window_samples = Int(round(abs(window[2] - window[1]) * fs))
    all_windows = []
    
    for event_time in event_times
        center_idx = argmin(abs.(time .- event_time))
        start_idx = center_idx + Int(round(window[1] * fs))
        end_idx = center_idx + Int(round(window[2] * fs))
        
        if start_idx >= 1 && end_idx <= length(signal)
            push!(all_windows, signal[start_idx:end_idx])
        end
    end
    
    if length(all_windows) == 0
        return plot(title="No valid event windows", 
                   size=figsize,
                   left_margin=20Plots.mm,
                   titlefontsize=16,
                   dpi=600)
    end
    
    # Average and SEM
    min_len = minimum(length.(all_windows))
    windows_matrix = hcat([w[1:min_len] for w in all_windows]...)
    
    avg_signal = vec(mean(windows_matrix, dims=2))
    sem_signal = vec(std(windows_matrix, dims=2)) ./ sqrt(size(windows_matrix, 2))
    
    # Time axis
    t_axis = range(window[1], window[2], length=min_len)
    
    # Plot with publication quality settings
    p = plot(t_axis, avg_signal,
            ribbon=sem_signal,
            fillalpha=0.3,
            fillcolor=RGB(0.5, 0.5, 0.5),
            linewidth=3,
            color=:black,
            xlabel="Time relative to event (s)",
            ylabel="Firing Rate (Hz) ± SEM",
            title="Event-Triggered Average (n=$(length(all_windows)) events)",
            legend=false,
            size=figsize,
            dpi=600,
            left_margin=20Plots.mm,
            right_margin=12Plots.mm,
            top_margin=12Plots.mm,
            bottom_margin=12Plots.mm,
            titlefontsize=16,
            guidefontsize=14,
            tickfontsize=12,
            framestyle=:box,
            grid=true,
            gridstyle=:dot,
            gridalpha=0.25,
            minorgrid=false)
    
    # Mark event time with thicker line
    vline!([0.0], 
          color=RGB(0.9, 0.1, 0.1), 
          linewidth=3, 
          alpha=0.8, 
          linestyle=:dash, 
          label="Event")
    
    return p
end

println("✓ Enhanced visualization module loaded")
println("  Publication-Quality Functions:")
println("    - plot_signal_with_events() - Signal + behavioral markers")
println("    - plot_spectrogram_with_events() - Spectrogram + markers")
println("    - create_summary_figure() - Comprehensive 3-panel figure")
println("    - plot_event_triggered_average() - Event-triggered average")
println("  ")
println("  Publication-Quality Settings:")
println("    - DPI: 600 (high resolution)")
println("    - Margins: 20mm left, 12mm others")
println("    - Fonts: 16pt titles, 14pt labels, 12pt ticks")
println("    - Line widths: 2.5-3pt")
println("    - Colors: High contrast RGB")
println("    - Grid: Subtle dots (0.25 alpha)")
println("  ")
println("  Ready for journal submission!")

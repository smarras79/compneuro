function neural_plots_scatter(x, y, ivar)

    plt = Plots.scatter(x, y;
                        markersize = 5,
                        color = :blue,
                        xlabel="true temporal distance",
                        ylabel="produced temporal distance",
                        title="mixture model",
                        titlefontsize = 24,
                        guidefontsize = 18,
                        legendfontsize = 14,
                        tickfontsize = 14,
                        legend = false,
                        size = (800, 600))
    
    Plots.plot!(plt, [-4, 4], [-4, 4], color=:black, linestyle=:solid)

    fout_name = string("neural_scatter", ivar, ".png")
    
    Plots.savefig(plt, string(fout_name))
    plt
    
    
end

function neural_plots(x, y1, y2, ivar, selected_filter)
    
    # Verify lengths match
    println("Length of time_bins: ", length(x))
    println("Length of fr3_smooth: ", length(y1))
    println("Length of fr4_smooth: ", length(y2))

    neural_plot = Plots.plot(x, y1, 
                             label="Condition 1 (4)",
                             line = (:blue, 2),
                             #marker = (:circle, 1, :blue),
                             grid=true,
                             xlabel="Time (s)",
                             ylabel="Firing Rate (Hz)",
                             legend=:topright,
                             titlefontsize = 22,
                             guidefontsize = 18,
                             legendfontsize = 14,
                             tickfontsize = 14,
                             size = (600, 400))

    Plots.plot!(neural_plot, x, y2,
                label="Condition 2 (5)",
                linewidth=2)

    # Uncomment for additional condition:
    # fr1_full = conv(vec(mean(fr1, dims=2)), kernel)
    # fr1_smooth = fr1_full[m:(end-(m-1))]
    # Plots.plot!(neural_plot, x, fr1_smooth, label="Condition 3 (2)", linewidth=2)

    # Set font size (approximate equivalent)
    #Plots.plot!(neural_plot, guidefontsize=15, tickfontsize=12, legendfontsize=12)

    # Save to file
    #savefig(neuprintln("Saved neural plot to: neural_firing_rates.png")
    
    fout_name = string("neural_curve", ivar, "_", selected_filter, ".png")
    Plots.savefig(neural_plot, string(fout_name))
    neural_plot
end

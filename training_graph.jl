using Statistics
using JLD2
using CairoMakie

file = jldopen("cor.jld2", "r")
cor_epochs, cormatrix_mean, cormatrix_std = file["cor_epochs"], file["cormatrix_mean"], file["cormatrix_std"]
close(file)

f = Figure()
Axis(f[1, 1],
title = "Pearson Correlation of Test Set",
xlabel = "epochs",
ylabel = "Pearson correlation")
errors = cormatrix_std[6,:]/sqrt(20)
errorbars!(cor_epochs, cormatrix_mean[6,:], 
            errors, errors, 
            whiskerwidth = 10,
            color = :blue,
            label = "Denoised - Predicted")
# plot position scatters so low and high errors can be discriminated
scatterlines!(cor_epochs, cormatrix_mean[6,:],
            markersize = 10,
             color = :blue,
             label = "Measured - Predicted")
lines!(cor_epochs, cormatrix_mean[4,:], color = :red)
axislegend()
f

# plot(cor_epochs, cormatrix_mean[4,:],
# label="test measure-pred")
# plot!(cor_epochs, cormatrix_mean[6,:],
#     label="test denois-pred",
#     xlabel="epochs",
#     ylabel="Pearson correlation",
#     marker = :circle,
#     yerror = cormatrix_std[6,:]./sqrt(20))
#plot!(cor_epochs, cormatrix_mean[5,:], label="test measure-denois")

save("cor_vs_epochs.png",f)
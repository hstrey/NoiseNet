using BDTools
using NIfTI
using CUDA
using Flux
using Statistics
using JLD2

#import data
groundtruthfile = "gt_clean.h5"
gt = BDTools.deserialize(groundtruthfile, GroundTruthCleaned)
ori64, _ = BDTools.Denoiser.standardize(reshape(gt.data[:,:,1],
    (size(gt.data[:,:,1])[1],1,size(gt.data[:,:,1])[2])))
sim64, _ = BDTools.Denoiser.standardize(reshape(gt.data[:,:,2],
    (size(gt.data[:,:,2])[1],1,size(gt.data[:,:,2])[2])))
ori = Float32.(ori64)
sim = Float32.(sim64)

N = 100
dev = Flux.cpu

# repeat training
cor_list10 = []
for i in 1:N
    model = DenoiseNet(BDTools.TrainParameters(; epochs = 10); dev = dev)

    losses, test_original, test_simulation = BDTools.train!(model, sim, ori)

    # calculate quality measures
    denoised_list = Array{Float32,1}[]
    for i in 1:size(ori)[3]
        denoised = BDTools.denoise(model, ori[:,:,i:i])
        push!(denoised_list, vec(denoised))
    end
    denois = vcat(denoised_list...)
    denoised_test_list = Array{Float32,1}[]
    for i in 1:size(test_original)[3]
        denoised_test = BDTools.denoise(model, test_original[:,:,i:i])
        push!(denoised_test_list, vec(denoised_test))
    end
    denois_test = vcat(denoised_test_list...)

    push!(cor_list10,(cor(vec(ori),vec(sim)),
                    cor(vec(ori),denois),
                    cor(vec(sim), denois),
                    cor(vec(test_original),vec(test_simulation)),
                    cor(vec(test_original),denois_test),
                    cor(vec(test_simulation),denois_test)))
end
cor10 = reduce(hcat,collect.(cor_list10))
cor10mean = mean(cor10,dims=2)
cor10std = std(cor10,dims=2)

cor_list15 = []
for i in 1:N
    model = DenoiseNet(BDTools.TrainParameters(; epochs = 15); dev = dev)

    losses, test_original, test_simulation = BDTools.train!(model, sim, ori)

    # calculate quality measures
    denoised_list = Array{Float32,1}[]
    for i in 1:size(ori)[3]
        denoised = BDTools.denoise(model, ori[:,:,i:i])
        push!(denoised_list, vec(denoised))
    end
    denois = vcat(denoised_list...)
    denoised_test_list = Array{Float32,1}[]
    for i in 1:size(test_original)[3]
        denoised_test = BDTools.denoise(model, test_original[:,:,i:i])
        push!(denoised_test_list, vec(denoised_test))
    end
    denois_test = vcat(denoised_test_list...)

    push!(cor_list15,(cor(vec(ori),vec(sim)),
                    cor(vec(ori),denois),
                    cor(vec(sim), denois),
                    cor(vec(test_original),vec(test_simulation)),
                    cor(vec(test_original),denois_test),
                    cor(vec(test_simulation),denois_test)))
end
cor15 = reduce(hcat,collect.(cor_list15))
cor15mean = mean(cor15,dims=2)
cor15std = std(cor15,dims=2)

cor_list20 = []
for i in 1:N
    model = DenoiseNet(BDTools.TrainParameters(; epochs = 20); dev = dev)

    losses, test_original, test_simulation = BDTools.train!(model, sim, ori)

    # calculate quality measures
    denoised_list = Array{Float32,1}[]
    for i in 1:size(ori)[3]
        denoised = BDTools.denoise(model, ori[:,:,i:i])
        push!(denoised_list, vec(denoised))
    end
    denois = vcat(denoised_list...)
    denoised_test_list = Array{Float32,1}[]
    for i in 1:size(test_original)[3]
        denoised_test = BDTools.denoise(model, test_original[:,:,i:i])
        push!(denoised_test_list, vec(denoised_test))
    end
    denois_test = vcat(denoised_test_list...)

    push!(cor_list20,(cor(vec(ori),vec(sim)),
                    cor(vec(ori),denois),
                    cor(vec(sim), denois),
                    cor(vec(test_original),vec(test_simulation)),
                    cor(vec(test_original),denois_test),
                    cor(vec(test_simulation),denois_test)))
end
cor20 = reduce(hcat,collect.(cor_list20))
cor20mean = mean(cor20,dims=2)
cor20std = std(cor20,dims=2)

cor_list50 = []
for i in 1:N
    model = DenoiseNet(BDTools.TrainParameters(; epochs = 50); dev = dev)

    losses, test_original, test_simulation = BDTools.train!(model, sim, ori)

    # calculate quality measures
    denoised_list = Array{Float32,1}[]
    for i in 1:size(ori)[3]
        denoised = BDTools.denoise(model, ori[:,:,i:i])
        push!(denoised_list, vec(denoised))
    end
    denois = vcat(denoised_list...)
    denoised_test_list = Array{Float32,1}[]
    for i in 1:size(test_original)[3]
        denoised_test = BDTools.denoise(model, test_original[:,:,i:i])
        push!(denoised_test_list, vec(denoised_test))
    end
    denois_test = vcat(denoised_test_list...)

    push!(cor_list50,(cor(vec(ori),vec(sim)),
                    cor(vec(ori),denois),
                    cor(vec(sim), denois),
                    cor(vec(test_original),vec(test_simulation)),
                    cor(vec(test_original),denois_test),
                    cor(vec(test_simulation),denois_test)))
end
cor50 = reduce(hcat,collect.(cor_list50))
cor50mean = mean(cor50,dims=2)
cor50std = std(cor50,dims=2)

cor_list100 = []
for i in 1:N
    model = DenoiseNet(BDTools.TrainParameters(; epochs = 100); dev = dev)

    losses, test_original, test_simulation = BDTools.train!(model, sim, ori)

    # calculate quality measures
    denoised_list = Array{Float32,1}[]
    for i in 1:size(ori)[3]
        denoised = BDTools.denoise(model, ori[:,:,i:i])
        push!(denoised_list, vec(denoised))
    end
    denois = vcat(denoised_list...)
    denoised_test_list = Array{Float32,1}[]
    for i in 1:size(test_original)[3]
        denoised_test = BDTools.denoise(model, test_original[:,:,i:i])
        push!(denoised_test_list, vec(denoised_test))
    end
    denois_test = vcat(denoised_test_list...)

    push!(cor_list100,(cor(vec(ori),vec(sim)),
                    cor(vec(ori),denois),
                    cor(vec(sim), denois),
                    cor(vec(test_original),vec(test_simulation)),
                    cor(vec(test_original),denois_test),
                    cor(vec(test_simulation),denois_test)))
end
cor100 = reduce(hcat,collect.(cor_list100))
cor100mean = mean(cor100,dims=2)
cor100std = std(cor100,dims=2)

cor_epochs = [10,15,20,50,100]
cormatrix_mean = reduce(hcat,[cor10mean,cor15mean,cor20mean,cor50mean,cor100mean])
cormatrix_std = reduce(hcat,[cor10std,cor15std,cor20std,cor50std,cor100std])

jldsave("cor2.jld2"; cor_epochs, cormatrix_mean, cormatrix_std)

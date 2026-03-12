

se<-read.table('/p/project1/hai_oneprot/bazarova1/oneprot-panda/seeds_for_pvalues.txt')

mode="AUC"

for (k in c(1:9)){
downstream=se[k,1]
print(downstream)
if (downstream=="Metal"){
  downstream="MetalIonBinding"
} else if (downstream=="Thermo"){
  downstream="ThermoStability"
}
#downstream='ThermoStability'
#downstream='MetalIonBinding'
model_names<-c("oneprot_full_allatom_no_seqsim_no_l1_A100_32900","oneprot_text_32900  ","oneprot_pocket_32900","oneprot_pocket_text_32900","oneprot_struct_graph_32900","oneprot_struct_graph_text_32900","oneprot_struct_graph_pocket_32900","oneprot_struct_graph_pocket_text_32900","oneprot_struct_token_32900","oneprot_struct_token_text_32900","oneprot_struct_token_pocket_32900","oneprot_struct_token_pocket_text_32900","oneprot_struct_token_struct_graph_32900","oneprot_struct_token_struct_graph_text_32900","oneprot_struct_token_struct_graph_pocket_32900","protrek_35          ","protrek             ","esm2                ","saprot              ","esm3                ","esmIF-new           ","embeddings_saprot   ")
model_samples=c()
if ((downstream=="HumanPPI")||(downstream=="EC")){

  model_names[21]="esmIF-08-25-2025    "
}

for (i in c(1:length(model_names))){
  print(model_names[i])
  seeds=se[k,i+1]
  seeds=as.numeric(strsplit(seeds[1], ",")[[1]])
  n=model_names[i]
  
  model_sample=c()
  for (s in seeds){
    #print(s)
    #ppi_seed=read.csv(paste0('/Users/alinabazarova/Downloads/no_sweep_', s, '/HumanPPI_MLP_results.csv'))
    ppi_seed=read.csv(paste0('/p/project1/hai_oneprot/bazarova1/oneprot-refined/results_checkpoints_no_sweep_',s,'/downstream_results/',downstream,'_MLP_results.csv'))
  
    
    
    # max_model=max(ppi_seed$valid_auc[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_auc)!=T)])
    # seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_auc==max_model),4]
    if (downstream=='ThermoStability'){
    max_model=max(ppi_seed$valid_spearman_rho[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_spearman_rho)!=T)])
    seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_spearman_rho==max_model),4]
    }

    if ((downstream %in% c('MetalIonBinding','HumanPPI','DeepLoc2','DeepLoc10'))&(mode!='AUC')) {
    max_model=max(ppi_seed$valid_accuracy[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_accuracy)!=T)])
    seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_accuracy==max_model),2]
    }

    if ((downstream %in% c('EC','GO-BP','GO-CC','GO-MF'))) {
      max_model=max(ppi_seed$valid_f1_max[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_f1_max)!=T)],na.rm=T)
      seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_f1_max==max_model),2]
    }

    if (downstream %in% c('MetalIonBinding','HumanPPI','DeepLoc2')&(mode=='AUC')) {
  max_model=max(ppi_seed$valid_auc[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_auc)!=T)])
  seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_auc==max_model),4]

  }
    # max_model=max(ppi_seed$valid_f1_max[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_f1_max)!=T)],na.rm=T)
    # #seed_sample=max(ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_f1_max==max_model),2],na.rm=T)
    # seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_f1_max==max_model),2]
    
    model_sample=c(model_sample,seed_sample)
  } 
    
    # max_esm=max(ppi_seed$valid_f1_micro[ppi_seed$model_type=='esm2                '])
    # max_oneprot=max(ppi_seed$valid_f1_micro[ppi_seed$model_type=='oneprot_struct_graph_pocket_text_32900'])
    #print(ppi_seed[(ppi_seed$model_type=='oneprot_struct_graph_pocket_text_32900')&(ppi_seed$valid_f1_micro==max_oneprot),])
    # print(cor(ppi_seed$valid_f1_micro[ppi_seed$model_type=='oneprot_struct_graph_pocket_text_32900'],ppi_seed$test_f1_micro[ppi_seed$model_type=='oneprot_struct_graph_pocket_text_32900']))
    # print(ppi_seed[(ppi_seed$model_type=='esm2                ')&(ppi_seed$valid_f1_micro==max_esm),])
    # print(cor(ppi_seed$valid_f1_micro[ppi_seed$model_type=='esm2                '],ppi_2$test_f1_micro[ppi_seed$model_type=='esm2                ']))
    
    
  
  #  model_means=c(model_means,mean(model_sample))
  #  model_stds=c(model_stds,sd(model_sample))
  if (length(seeds)<6){
  ppi_seed=read.csv(paste0('/p/project1/hai_oneprot/bazarova1/oneprot-refined/results_checkpoints/downstream_results/',downstream,'_MLP_results.csv'))
  # 
  if (downstream=="ThermoStability"){
  max_model=max(ppi_seed$valid_spearman_rho[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_spearman_rho)!=T)])
   seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_spearman_rho==max_model),4]
  }

  if (downstream %in% c('MetalIonBinding','HumanPPI','DeepLoc2','DeepLoc10')&(mode!='AUC')) {
  max_model=max(ppi_seed$valid_accuracy[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_accuracy)!=T)])
  seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_accuracy==max_model),2]
  }

  if (downstream %in% c('EC','GO-BP','GO-CC','GO-MF')) {
    max_model=max(ppi_seed$valid_f1_max[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_f1_max)!=T)],na.rm=T)
    seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_f1_max==max_model),2]
  }
  # max_model=max(ppi_seed$valid_f1_max[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_f1_max)!=T)],na.rm=T)
  # #seed_sample=max(ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_f1_max==max_model),2],na.rm=T)
  # seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_f1_max==max_model),2]
  #
  if (downstream %in% c('MetalIonBinding','HumanPPI','DeepLoc2')&(mode=='AUC')) {
  max_model=max(ppi_seed$valid_auc[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_auc)!=T)])
  seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_auc==max_model),4]

  }
  model_sample=c(model_sample,seed_sample)
  } else if ((n=="oneprot_struct_token_text_32900")&(downstream=="GO-MF")){
    
    ppi_seed=read.csv(paste0('/p/project1/hai_oneprot/bazarova1/oneprot-refined/results_checkpoints/downstream_results/',downstream,'_MLP_results.csv'))
    # 
  if (downstream=="ThermoStability"){
  max_model=max(ppi_seed$valid_spearman_rho[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_spearman_rho)!=T)])
   seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_spearman_rho==max_model),4]
  }

  if (downstream %in% c('MetalIonBinding','HumanPPI','DeepLoc2','DeepLoc10')&(mode!='AUC')) {
  max_model=max(ppi_seed$valid_accuracy[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_accuracy)!=T)])
  seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_accuracy==max_model),2]
  }

  if (downstream %in% c('EC','GO-BP','GO-CC','GO-MF')) {
    max_model=max(ppi_seed$valid_f1_max[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_f1_max)!=T)],na.rm=T)
    seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_f1_max==max_model),2]
  }
  # max_model=max(ppi_seed$valid_f1_max[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_f1_max)!=T)],na.rm=T)
  # #seed_sample=max(ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_f1_max==max_model),2],na.rm=T)
  # seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_f1_max==max_model),2]
  #
  if (downstream %in% c('MetalIonBinding','HumanPPI','DeepLoc2')&(mode=='AUC')) {
  max_model=max(ppi_seed$valid_auc[(ppi_seed$model_type==n)&(is.nan(ppi_seed$test_auc)!=T)])
  seed_sample=ppi_seed[(ppi_seed$model_type==n)&(ppi_seed$valid_auc==max_model),4]
    # 
  }
    model_sample=c(model_sample,rep(seed_sample,4))
    
  }
  
  if (n=="saprot              "){
    
   if (downstream=="DeepLoc2"){
     model_sample=c(model_sample,rep(0.913,4))
   } else if(downstream=="DeepLoc10"){
     
     model_sample=c(model_sample,rep(0.795,4))
     
   } else if (downstream=="EC"){
     
     model_sample=c(model_sample,rep(0.862,4))
   
     } else if (downstream=="GO-BP"){
     
     model_sample=c(model_sample,rep(0.467,4))
   
     } else if (downstream=="GO-CC"){
     
     model_sample=c(model_sample,rep(0.548,4))
     } else if (downstream=="GO-MF"){
     
       model_sample=c(model_sample,rep(0.613,4))
     } else if (downstream=="HumanPPI"){
     
       model_sample=c(model_sample,rep(0.863,4))
       
     } else if (downstream=="MetalIonBinding"){
     
       model_sample=c(model_sample,rep(0.689,4))
       
     } else if (downstream=="ThermoStability"){
     
       model_sample=c(model_sample,rep(0.705,4))
   }
    
    
  }
  
  model_samples[[i]]<-model_sample[seq(1,length(model_sample),4)]
  #print(mean(model_samples[[i]],na.rm=T))
  #print(sd(model_samples[[i]],na.rm=T))
  
}



pvalues2<-array(NA,c(16,16))
for (i in c(1:15,18)){
  
  for (j in c(1:15,18)){
    # if (mean(model_samples[[i]])>mean(model_samples[[j]])){
    #   
    #   alt=c("greater")
    #   
    # } else{
    #   
    #   alt=c("less")
    # }
    alt=c("two.sided")
    a<-wilcox.test(model_samples[[i]],model_samples[[j]],alternative=alt)
   
    #pvalues2[i,j-15]=a$p.value
    if (i==18){
      if (j==18){
        pvalues2[16,16]=a$p.value
      } else{
        pvalues2[16,j]=a$p.value
      }
    } else{
      if (j==18){
        pvalues2[i,16]=a$p.value
      } else{
        pvalues2[i,j]=a$p.value
      }
    }
    #pvalues2[i,j]=a$p.value
     
  }
}



names<-c('OneProt-5','Text','Pocket','Pocket+Text','SG','SG+Text','SG+Pocket','OneProt-4','ST','ST+Text','ST+pocket','ST+Pocket+Text','ST+SG','ST+SG+Text','SG+ST+Pocket','ProTrek-35M','ProTrek-650M','ESM2','SaProt','ESM3','ESM-IF','OpenFold')
rownames(pvalues2)<-names[c(1:15,18)]
colnames(pvalues2)<-names[c(1:15,18)]

if (mode!='AUC'){
  write.csv(pvalues2,paste0("/p/project1/hai_oneprot/bazarova1/oneprot-panda/pvalues/",downstream,"_2p_ablations.csv"))
} else {
  write.csv(pvalues2,paste0("/p/project1/hai_oneprot/bazarova1/oneprot-panda/pvalues/",downstream,"_2p_AUC_ablations.csv"))
}



pvalues<-array(NA,c(16,16))
for (i in c(1:15,18)){
  for (j in c(1:15,18)){
  #for (j in c(16:22)){
    # if (mean(model_samples[[i]])>mean(model_samples[[j]])){
    #   
    #   alt=c("greater")
    #   
    # } else{
    #   
    #   alt=c("less")
    # }
    alt=c("greater")
    a<-wilcox.test(model_samples[[i]],model_samples[[j]],alternative=alt)
   
    #pvalues[i,j-15]=a$p.value
    if (i==18){
      if (j==18){
        pvalues[16,16]=a$p.value
      } else{
        pvalues[16,j]=a$p.value
      }
    } else{
      if (j==18){
        pvalues[i,16]=a$p.value
      } else{
        pvalues[i,j]=a$p.value
      }
    }
     
  }
}

names<-c('OneProt-5','Text','Pocket','Pocket+Text','SG','SG+Text','SG+Pocket','OneProt-4','ST','ST+Text','ST+pocket','ST+Pocket+Text','ST+SG','ST+SG+Text','SG+ST+Pocket','ProTrek-35M','ProTrek-650M','ESM2','SaProt','ESM3','ESM-IF','OpenFold')
rownames(pvalues)<-names[c(1:15,18)]
#colnames(pvalues)<-names[16:22]
colnames(pvalues)<-names[c(1:15,18)]

if (mode!='AUC'){
  write.csv(pvalues,paste0("/p/project1/hai_oneprot/bazarova1/oneprot-panda/pvalues/",downstream,"_p_ablations.csv"))
} else {
 write.csv(pvalues,paste0("/p/project1/hai_oneprot/bazarova1/oneprot-panda/pvalues/",downstream,"_p_AUC_ablations.csv"))
}
#}
}
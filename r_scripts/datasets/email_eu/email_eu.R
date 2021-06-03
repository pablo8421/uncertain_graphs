############################
# 
# Cora dataset exploration
# https://paperswithcode.com/dataset/cora
# Scientific publications classified into one of seven classes.
# Pablo Javier Sánchez Díaz
# 
############################

# Clean working space
rm(list = ls())

# Loading used libraries
library(tidyverse)
library(igraph)
library(GGally)
library(stats)



# base path
datsets_location <- 'D:/Pablo/clases/UJM/2. Semester, 2021/Mining Uncertain Social Networks/Repository/r_scripts/datasets/'

setwd(paste(datsets_location,'email_eu/',sep = ''))
dataset_name <- 'email_eu'
dataset_color <- '#21cbc'


# Loading data
email_edges <- read.csv(file='email-Eu-core.txt', sep = ' ', header=FALSE)
email_labels <- read.csv(file='email-Eu-core-department-labels.txt', sep = ' ', header=FALSE)

email_labels$V2 <- email_labels$V2 +1


### Graph creation ###


# Creating the graph object
net <- graph_from_data_frame(d=email_edges, vertices=email_labels, directed=FALSE) 
# To avoid multiplex graphs
net <- simplify(net)

# Calculationg
num_vertex <- length(V(net))
num_edges <- length(E(net))

## The three metrics that define it as a social network ##

#https://igraph.org/r/doc/diameter.html
net_density <- graph.density(net)
#Low density defined as O(m) << O(n^2)

#Degree distribution
vertex_degrees <- as.data.frame(table(degree(net))) %>% rename(degree = Var1)

#https://igraph.org/r/doc/diameter.html
net_diameter <- diameter(net)

## End of the metrics that define it as a social network ##

#https://igraph.org/r/doc/modularity.igraph.html
net_modularity <- modularity(net, as_data_frame(net, what='vertices')$V2)


#https://igraph.org/r/doc/transitivity.html
#https://stackoverflow.com/questions/48853610/average-clustering-coefficient-of-a-network-igraph
clustering_coefficient <- transitivity(net, type = 'average')


# What to do with this?
Connected_components <- components(net)

#https://igraph.org/r/doc/betweenness.html
net_betweenness <- mean(betweenness(net))

# Vertex betweenness
vertext_betweennes <- as.data.frame(table(betweenness(net))) %>%
  rename(betweenness = Var1)
vertext_betweennes$betweenness <- as.numeric(vertext_betweennes$betweenness)
vertext_betweennes$betweenness <- (vertext_betweennes$betweenness - min(vertext_betweennes$betweenness))/(max(vertext_betweennes$betweenness)-min(vertext_betweennes$betweenness))



#https://igraph.org/r/doc/closeness.html
net_closeness <- mean(closeness(net))

#https://igraph.org/r/doc/components.html
connected_components <- count_components(net)

# Visualization
# Source: https://briatte.github.io/ggnet/
ggnet2(net, node.size = 2, color = 'label', edge.size = 1, edge.color = 'grey', palette = 'Set1')


#Plotting betweenness
# Plotting Degrees 
vertext_betweennes %>%
  mutate(betweenness=round(betweenness, 1)) %>%
  group_by(betweenness) %>%
  summarise(Freq=sum(Freq)) %>%
  ggplot( aes(x=betweenness, y=Freq)) +
  geom_line(group=1, color="black") +
  geom_point(shape=21, color="black", fill="#1DA1F2", size=5) +
  ggtitle(paste(dataset_name,"Betweenness"))



#Plotting everything!
# Plotting Degrees 
vertex_degrees %>%
  ggplot( aes(x=degree, y=Freq)) +
  geom_line(group=1, color="black") +
  geom_point(shape=21, color="black", fill="#1DA1F2", size=5) +
  annotate("text", 
           x = max(as.numeric(vertex_degrees$degree))-max(as.numeric(vertex_degrees$degree))/3, 
           y = max(vertex_degrees$Freq)-(max(vertex_degrees$Freq)/6),
           hjust = 0,
           label = paste(paste('Nodes:',format(num_vertex,  big.mark = ',')),
                         paste('Edges:',format(num_edges,  big.mark = ',')),
                         paste('Diameter:',net_diameter),
                         paste('Density:',format(net_density, nsmall = 6)),
                         paste('Modularity:',format(net_modularity, nsmall = 2)),
                         paste('Clustering Coeff:',format(clustering_coefficient, nsmall = 4)),
                         paste('Betweeness:',format(net_betweenness, nsmall = 2)),
                         paste('Closeness:',format(net_closeness, nsmall = 2)),
                         paste('Connected Components:',connected_components),
                         sep='\n')) +
  ggtitle(paste(dataset_name,"Graph"))



#Save graph
#https://igraph.org/r/doc/write_graph.html
write_graph(net, 'cora_graph.txt', format='pajek')

#Read graph
#https://igraph.org/r/doc/read_graph.html
net <- read_graph('cora_graph.txt', format='pajek')


# Save a general summary of all
datasets_summary <- read.csv(file=paste(datsets_location,'datasets_summary.csv',sep=''), sep = ',')
datasets_summary <- datasets_summary %>% filter(Name!=dataset_name) 

datasets_summary <- datasets_summary %>% add_row(
  Name=dataset_name,
  Nodes=num_vertex,
  Edges=num_edges,
  Diameter=net_diameter,
  Density=net_density,
  Modularity=net_modularity,
  Clustering.Coefficient=clustering_coefficient,
  Betweeness=net_betweenness,
  Closeness=net_closeness,
  Connected.Components=connected_components,
  Communities=length(unique(as_data_frame(net, what='vertices')$V2))
)

write.csv(datasets_summary, file=paste(datsets_location,'datasets_summary.csv',sep=''), row.names=FALSE)


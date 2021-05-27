############################
# 
# Dancer generated graphs exploration
# https://perso.univ-st-etienne.fr/largeron/DANCer_Generator/
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
datsets_location <- 'D:/Pablo/clases/UJM/2. Semester, 2021/Mining Uncertain Social Networks/Graphs Visualization/datasets/'

setwd(paste(datsets_location,'dancer/second/',sep = ''))
dataset_name <- 'Dancer 02'
dataset_color <- '#2b9469'


# Loading data
graph_data <- file('t9.graph', open='r')

vertices <- data.frame(id=character(),
                       atributes=character(),
                       community=character())


edges <- data.frame(from=character(),
                    to=character())

in_vertices = TRUE;
while (length(one_line <- readLines(graph_data, n = 1, warn = FALSE)) > 0) {
  if (one_line == '#')
  {
    in_vertices = FALSE;
  }
  
  if(one_line  != '# Vertices' && one_line  != '# Edges' && one_line  != '#')
  {
    if(in_vertices)
    {
      line = as.list(strsplit(one_line, ';')[[1]])
      
      vertices[nrow(vertices) + 1,] = c(line[])
    }
    else
    {
      line = as.list(strsplit(one_line, ';')[[1]])
      
      edges[nrow(edges) + 1,] = c(line[])
    }
  }
  
}

edges$from <- as.numeric(edges$from)
edges$to <- as.numeric(edges$to)

vertices$id <- as.numeric(vertices$id)
vertices$community <- as.numeric(vertices$community) + 1
vertices <- vertices %>% select(id,community)


### Graph creation ###

# Creating the graph object
net <- graph_from_data_frame(d=edges, vertices=vertices, directed=FALSE) 
# To avoid multiplex graphs
#net <- simplify(net)

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
net_modularity <- modularity(net, vertices$community)

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
write_graph(net, paste(dataset_name,'_graph.txt',sep=''), format='pajek')

#Read graph
#https://igraph.org/r/doc/read_graph.html
net <- read_graph(paste(dataset_name,'_graph.txt',sep=''), format='pajek')


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
  Communities=length(vertices$community)
)

write.csv(datasets_summary, file=paste(datsets_location,'datasets_summary.csv',sep=''), row.names=FALSE)

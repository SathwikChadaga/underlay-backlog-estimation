import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Network:
  def __init__(
    self,
    num_tunnels,
    num_nodes,
    num_edges,
    underlay_service_rates,
    external_arrival_rates,
    tunnel_edge2node_adjacencies
  ):
    self.num_tunnels = num_tunnels
    self.num_nodes = num_nodes
    self.num_edges = num_edges
    self.underlay_service_rates = underlay_service_rates
    self.external_arrival_rates = external_arrival_rates
    self.tunnel_edge2node_adjacencies = tunnel_edge2node_adjacencies
    self.time = 0

  def reset(self):
    self.queue_backlogs = np.zeros([self.num_tunnels, self.num_nodes,])
    self.time = 0
    return self.queue_backlogs, self.time

  def act(self):
    return

  def step(self, overlay_offered_rates):

    # handle each tunnel as though it is a different network
    for tunnel_index in range(self.num_tunnels):
      # self.queue_backlogs[tunnel_index,:]: Individual queue backlogs along tunnel_index (underlay and overlay)
      # external_arrivals: External arrivals 
      # all_offered_rates: Offered rates by each edges
      # actual_flows: Actual packets served by each edges <= offered rates

      # external arrivals
      external_arrivals = np.random.poisson(self.external_arrival_rates[tunnel_index,:], self.num_nodes)
      self.queue_backlogs[tunnel_index,:] += external_arrivals

      # flows along underlay edges
      underlay_offered_rates_tunnel_index = np.random.poisson(self.underlay_service_rates[tunnel_index,:], self.num_edges)

      # flows along all edges
      all_offered_rates = underlay_offered_rates_tunnel_index + overlay_offered_rates[tunnel_index,:]

      # choose min(current queue size at source of the edge, offered rate of the edge)
      # this code assumes that there is no branching within a tunnel
      actual_flows = np.min(np.vstack(((self.tunnel_edge2node_adjacencies[tunnel_index,:,:].T < 0)@self.queue_backlogs[tunnel_index,:], all_offered_rates)), axis=0)

      # apply flows
      self.queue_backlogs[tunnel_index,:] += self.tunnel_edge2node_adjacencies[tunnel_index,:,:]@actual_flows
      self.queue_backlogs[tunnel_index, self.queue_backlogs[tunnel_index,:] < 0] = 0

      # the destination queue is a sink
      self.queue_backlogs[tunnel_index, -1] = 0

    self.time += 1
    return self.queue_backlogs, self.time
  
  def simulate(self, overlay_service_rates, total_time, custom_seed = None):
    # reset network
    if(custom_seed != None): np.random.seed(custom_seed)
    queue_backlogs, time = self.reset()

    # get network parameters
    num_nodes = self.num_nodes
    num_edges = self.num_edges
    num_tunnels = self.num_tunnels
    tunnel_edge2node_adjacencies = self.tunnel_edge2node_adjacencies

    # initial variables to save results
    packets_in_flight = np.zeros([total_time, num_tunnels]) # packets in flight in each tunnel
    tunnel_backlogs = np.zeros([total_time, num_tunnels]) # total queue backlogs in each tunnel
    queue_backlogs = np.zeros([total_time, num_nodes]) # backlogs in each queue

    # simulate for total_time iterations
    queue_backlogs[time, : ] = np.sum(queue_backlogs, axis=0)
    while(time < total_time-1):
        # get overlay rates and simulate one time step
        overlay_offered_rates = np.random.poisson(overlay_service_rates, [num_tunnels, num_edges, ])
        queue_backlogs_per_tunnel, time = self.step(overlay_offered_rates)

        # save relevant information
        queue_backlogs[time, : ] = np.sum(queue_backlogs_per_tunnel, axis=0)
        packets_in_flight[time, :] = np.sum(queue_backlogs_per_tunnel, axis=1)

    for tunnel_ind in range(num_tunnels):
        is_queue_in_tunnel = (np.sum(tunnel_edge2node_adjacencies[tunnel_ind,:,:] == -1, axis=1) == 1)
        tunnel_backlogs[:, tunnel_ind] = np.sum(queue_backlogs[:, is_queue_in_tunnel], axis=1)

    return packets_in_flight, tunnel_backlogs
  
  def visualize(self):
    G = nx.DiGraph()

    color_list = ['r', 'b', 'g', 'y', 'm']
    for tunnel_ind in range(self.num_tunnels):
        adjacency_matrix = self.tunnel_edge2node_adjacencies[tunnel_ind,:,:].T
        tunnel_color = color_list[tunnel_ind]

        for row in adjacency_matrix:
            i = np.argwhere(row == -1)
            j = np.argwhere(row == 1)
            if(i.shape[0] == 0): continue
            G.add_edge(i[0][0], j[0][0],color=tunnel_color)  

    # Visualize the network
    pos = nx.spring_layout(G) 
    colors = nx.get_edge_attributes(G,'color').values()
    nx.draw_networkx(G, pos, with_labels=True, edge_color=colors, node_size=500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", arrowsize=20, width=2)

    plt.title("Network Visualization")
    plt.show()

import numpy as np

class Network:
  def __init__(
    self,
    k_tunnels,
    n_nodes,
    m_edges,
    underlay_service_rates,
    external_arrival_rates,
    tunnel_edge2node_adjacencies
  ):
    self.k_tunnels = k_tunnels
    self.n_nodes = n_nodes
    self.m_edges = m_edges
    self.underlay_service_rates = underlay_service_rates
    self.external_arrival_rates = external_arrival_rates
    self.tunnel_edge2node_adjacencies = tunnel_edge2node_adjacencies
    self.time = 0

  def reset(self):
    self.queue_backlogs = np.zeros([self.k_tunnels, self.n_nodes,])
    self.time = 0
    return self.queue_backlogs, self.time

  def act(self):
    return

  def step(self, overlay_offered_rates):

    # handle each tunnel as though it is a different network
    for tt in range(self.k_tunnels):
      # if(tt == 1):
      #   print('t = ' + str(self.time))
      #   print('Q(t) = ' + str(self.queue_backlogs[tt,:]))

      # external arrivals
      external_arrivals = np.random.poisson(self.external_arrival_rates[tt,:], self.n_nodes)
      self.queue_backlogs[tt,:] += external_arrivals

      # flows along edges
      all_offered_rates = np.random.poisson(self.underlay_service_rates[tt,:], self.m_edges) + overlay_offered_rates[tt,:]

      # make sure flows are not greater than current queue size
      actual_rates = np.min(np.vstack(((self.tunnel_edge2node_adjacencies[tt,:,:].T < 0)@self.queue_backlogs[tt,:], all_offered_rates)), axis=0)

      # apply flows
      self.queue_backlogs[tt,:] += self.tunnel_edge2node_adjacencies[tt,:,:]@actual_rates
      self.queue_backlogs[tt, self.queue_backlogs[tt,:] < 0] = 0
      # if(tt == 1):
      #   print('External arrivals = ' + str(external_arrivals))
      #   print('Offered rates = ' + str(all_offered_rates))
      #   print('Actual rates = ' + str(actual_rates))
      #   print('Q(t+1) = ' + str(self.queue_backlogs[tt,:]))
      #   print(' ')

      # the destination queue is a sink
      self.queue_backlogs[tt, -1] = 0

    self.time += 1
    return self.queue_backlogs, self.time
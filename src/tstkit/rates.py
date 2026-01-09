import numpy as np

class RatesTable:
    def __init__(self, n_events, equal_events,
                 site_symbol_ranges, temperature_data, T):
        self.n_sites = len(equal_events)
        self.max_events = max(n_events)

        self.rates = np.zeros((self.n_sites, self.max_events), dtype=np.float64)
        self.total_rate = np.zeros(self.n_sites, dtype=np.float64)

        self._build(equal_events, site_symbol_ranges, temperature_data[T])

    def _build(self, equal_events, site_symbol_ranges, temp_data_T):
        for site_type, site_range in site_symbol_ranges.items():
            base_rates = temp_data_T.rates[site_type]

            for site in range(site_range[0], site_range[1] + 1):
                rate_no = 0
                for idx, equal_count in enumerate(equal_events[site]):
                    rate = base_rates[idx]
                    for _ in range(equal_count):
                        self.rates[site, rate_no] = rate
                        self.total_rate[site] += rate
                        rate_no += 1
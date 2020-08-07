# %% Preamble
import unittest
import src.lobtools as lt


# %% Testing Class
class testOrderBook(unittest.TestCase):

    def initialize_book(self, lob):
        self.assertEqual(lob.submit_limitorder(-1, 105, 100, 0.05), 0)
        self.assertEqual(lob.submit_limitorder(-1, 107.5, 200, 0.06), 1)
        self.assertEqual(lob.submit_limitorder(1, 95, 200, 0.07), 2)
        self.assertEqual(lob.submit_limitorder(1, 97.5, 400, 0.08), 3)
        self.assertEqual(lob.submit_limitorder(1, 95, 100, 0.09), 4)
        self.assertEqual(lob.submit_limitorder(-1, 105, 50, 0.10), 5)

    def test_submit_limitorder(self):
        lob = lt.OrderBook('XYZ', tick_size=0.01)
        # Check Book Is Empty
        self.assertEqual(lob.buyside_volume(), 0)
        self.assertEqual(lob.sellside_volume(), 0)

        # Check Non-Executable Submissions
        self.initialize_book(lob)

        # Check State of Book is as expected
        self.assertEqual(lob.buyside_volume(), 700)
        self.assertEqual(lob.sellside_volume(), 350)
        self.assertEqual(lob.bestbid_volume(), 400)
        self.assertEqual(lob.bestask_volume(), 150)
        self.assertEqual(lob.spread(), 7.5)

        # Check Executable Limit Order
        self.assertEqual(lob.submit_limitorder(1, 110, 50, 0.11), 6)
        self.assertEqual(lob.submit_limitorder(-1, 90, 450, 0.12), 7)

        # Check Correct order executions
        self.assertEqual(lob.limit_orders[0][4], 'active')
        self.assertEqual(lob.limit_orders[0][2], 50)
        self.assertEqual(lob.limit_orders[3][4], 'executed')
        self.assertEqual(lob.limit_orders[3][2], 0)
        self.assertEqual(lob.limit_orders[2][4], 'active')
        self.assertEqual(lob.limit_orders[2][2], 150)

        # Check State of Book is as expected
        self.assertEqual(250, lob.buyside_volume())
        self.assertEqual(300, lob.sellside_volume())
        self.assertEqual(250, lob.bestbid_volume())
        self.assertEqual(100, lob.bestask_volume())
        self.assertEqual(10, lob.spread())

        # Check that executable limit orders leave behind volume
        self.assertEqual(8, lob.submit_limitorder(1, lob.bestask(), 105, 0.13))
        self.assertEqual(200, lob.sellside_volume())
        self.assertEqual(255, lob.buyside_volume())
        self.assertEqual(5, lob.bestbid_volume())



    def test_submit_marketorder(self):
        lob = lt.OrderBook('XYZ', tick_size=0.01)

        # Initialize Book
        self.initialize_book(lob)

        # Test Market Orders
        self.assertEqual(lob.submit_marketorder(1, 100, 0.11, True), 10500)
        self.assertEqual(lob.submit_marketorder(-1, 100, 0.12, True), 9750)

        # Check State of Book is as expected
        self.assertEqual(lob.buyside_volume(), 600)
        self.assertEqual(lob.sellside_volume(), 250)
        self.assertEqual(lob.bestbid_volume(), 300)
        self.assertEqual(lob.bestask_volume(), 50)
        self.assertEqual(lob.spread(), 7.5)


if __name__ == '__main__':
    unittest.main()  # allows us to run directly from IDE instead of long comand through terminal

# %%

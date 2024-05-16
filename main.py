from streamlit.testing import AppTest
import your_app
class MyAppTest(AppTest):
    def test_search_success(self):
        self.run_app(your_app)
        self.type("text_input", "your query")
        self.click("button")
        self.wait_for("image")  # Wait for the first image to load
        # Assert expected results
        self.assert_text("Image description")
if __name__ == "__main__":
    MyAppTest().run()
// Function to fetch heart attack prediction data
function importData() {
  console.log('Importing data...'); // Check if the function is being called
  fetch('http://localhost:5000/importData')
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.text();
    })
    .then(data => {
      console.log('Data received:', data); // Log the response from the server
      // Perform further actions if needed
    })
    .catch(error => {
      console.error('There was a problem with the fetch operation:', error);
    });
}

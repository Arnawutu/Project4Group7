// Define the JavaScript function for form validation
function validateForm() {
  // Get input values
  var age = document.getElementById("age").value;
  var cholesterol = document.getElementById("cholesterol").value;
  var heartRate = document.getElementById("heart_rate").value;
  var diabetes = document.getElementById("diabetes").value;
  var familyHistory = document.getElementById("family_history").value;
  var smoking = document.getElementById("smoking").value;
  var obesity = document.getElementById("obesity").value;
  var alcoholConsumption = document.getElementById("alcohol_consumption").value;
  var previousHeartProblems = document.getElementById("previous_heart_problems").value;
  var medicationUse = document.getElementById("medication_use").value;
  var bmi = document.getElementById("bmi").value;
  var triglycerides = document.getElementById("triglycerides").value;
  var systolicPressure = document.getElementById("systolic_pressure").value;
  var diastolicPressure = document.getElementById("diastolic_pressure").value;

  // Perform validation for age
  if (age < 0 || age > 130 || isNaN(age)) {
      alert("Please enter a valid age.");
      return false; // Prevent form submission
  }

  // Perform validation for cholesterol
  if (cholesterol < 0 || isNaN(cholesterol)) {
      alert("Please enter a valid cholesterol level.");
      return false; // Prevent form submission
  }

  // Perform validation for heart rate
  if (heartRate < 0 || isNaN(heartRate)) {
      alert("Please enter a valid heart rate.");
      return false; // Prevent form submission
  }

  // Perform validation for diabetes (0 or 1)
  if (diabetes !== '0' && diabetes !== '1') {
      alert("Please select a valid value for diabetes.");
      return false; // Prevent form submission
  }

  // Perform validation for family history (0 or 1)
  if (familyHistory !== '0' && familyHistory !== '1') {
      alert("Please select a valid value for family history.");
      return false; // Prevent form submission
  }

  // Perform validation for smoking (0 or 1)
  if (smoking !== '0' && smoking !== '1') {
      alert("Please select a valid value for smoking.");
      return false; // Prevent form submission
  }

  // Perform validation for obesity (0 or 1)
  if (obesity !== '0' && obesity !== '1') {
      alert("Please select a valid value for obesity.");
      return false; // Prevent form submission
  }

  // Perform validation for alcohol consumption (0, 1, or 2)
  if (alcoholConsumption !== '0' && alcoholConsumption !== '1' && alcoholConsumption !== '2') {
      alert("Please select a valid value for alcohol consumption.");
      return false; // Prevent form submission
  }

  // Perform validation for previous heart problems (0 or 1)
  if (previousHeartProblems !== '0' && previousHeartProblems !== '1') {
      alert("Please select a valid value for previous heart problems.");
      return false; // Prevent form submission
  }

  // Perform validation for medication use (0 or 1)
  if (medicationUse !== '0' && medicationUse !== '1') {
      alert("Please select a valid value for medication use.");
      return false; // Prevent form submission
  }

  // Perform validation for BMI
  if (bmi < 0 || isNaN(bmi)) {
      alert("Please enter a valid BMI.");
      return false; // Prevent form submission
  }

  // Perform validation for triglycerides
  if (triglycerides < 0 || isNaN(triglycerides)) {
      alert("Please enter a valid triglycerides level.");
      return false; // Prevent form submission
  }

  // Perform validation for systolic pressure
  if (systolicPressure < 0 || isNaN(systolicPressure)) {
      alert("Please enter a valid systolic pressure.");
      return false; // Prevent form submission
  }

  // Perform validation for diastolic pressure
  if (diastolicPressure < 0 || isNaN(diastolicPressure)) {
      alert("Please enter a valid diastolic pressure.");
      return false; // Prevent form submission
  }

  // If all validations pass, return true to allow form submission
  return true;
}

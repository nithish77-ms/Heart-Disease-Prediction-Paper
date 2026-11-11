export const clinicalGuidance = {
  age: {
    label: 'Age',
    clinicalRange: 'No single age diagnoses heart disease.',
    clinicalNote: 'Heart disease risk generally rises with age.',
    reason: 'Age is one part of heart-health risk. It does not determine a person’s heart health on its own.',
    source: 'CDC: Heart disease risk factors',
    sourceUrl: 'https://www.cdc.gov/heart-disease/risk-factors/index.html',
  },
  sex: {
    label: 'Sex recorded in the source data',
    clinicalRange: 'Model choices: Male or Female.',
    clinicalNote: 'Choose the category in your source medical record. These model categories are not a diagnosis or a complete description of sex.',
    reason: 'This is a background detail included in the model; by itself, it cannot tell whether heart disease is present.',
    source: 'UCI: Heart Disease dataset fields',
    sourceUrl: 'https://archive.ics.uci.edu/dataset/45/heart+disease',
  },
  cp: {
    label: 'Chest-pain pattern',
    clinicalRange: 'No single normal/abnormal range.',
    clinicalNote: 'Choose the description that best matches the source record.',
    reason: 'Some chest-pain patterns can happen when the heart muscle gets less blood. Chest pain should be assessed by a health professional.',
    source: 'American Heart Association: Angina',
    sourceUrl: 'https://www.heart.org/en/health-topics/heart-attack/angina-chest-pain',
  },
  trestbps: {
    label: 'Resting systolic blood pressure',
    clinicalRange: 'Generally normal: below 120 mmHg systolic. High blood pressure: 130 mmHg systolic or 80 mmHg diastolic and above.',
    clinicalNote: 'This model records systolic pressure only; the complete reading includes both numbers.',
    reason: 'High blood pressure can strain the heart and blood vessels over time.',
    source: 'American Heart Association: Blood pressure readings',
    sourceUrl: 'https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings',
  },
  chol: {
    label: 'Total cholesterol',
    clinicalRange: 'About 150 mg/dL is optimal; above 200 mg/dL may be high.',
    clinicalNote: 'A clinician interprets cholesterol with the full lipid panel and personal health history.',
    reason: 'Higher cholesterol can contribute to fatty build-up in blood vessels over time.',
    source: 'CDC: About cholesterol',
    sourceUrl: 'https://www.cdc.gov/cholesterol/about/index.html',
  },
  fbs: {
    label: 'Fasting blood sugar',
    clinicalRange: 'Fasting lab: 99 mg/dL or below is normal; 100–125 mg/dL is the prediabetes range; 126 mg/dL or above is in the diabetes range.',
    clinicalNote: 'This model asks whether fasting blood sugar is above 120 mg/dL. It is not a diabetes diagnosis; a clinician usually confirms a result with another test.',
    reason: 'High blood sugar over time can damage blood vessels and raise heart-health risk.',
    source: 'NIDDK: Diabetes tests and diagnosis',
    sourceUrl: 'https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis',
  },
  restecg: {
    label: 'Resting ECG pattern',
    clinicalRange: 'No single normal range for these model categories.',
    clinicalNote: 'Choose the pattern recorded on the ECG report; a clinician interprets the result.',
    reason: 'An ECG pattern can give clinicians clues about the heart’s electrical signals or muscle; it does not confirm heart disease by itself.',
    source: 'American Heart Association: Electrocardiogram (ECG or EKG)',
    sourceUrl: 'https://www.heart.org/en/health-topics/heart-attack/diagnosing-a-heart-attack/electrocardiogram-ecg-or-ekg',
  },
  thalach: {
    label: 'Maximum heart rate during exercise',
    clinicalRange: 'A rough exercise guide estimates maximum heart rate as 220 minus age.',
    clinicalNote: 'This is not a heart-disease cutoff. Heart rate during exercise depends on age, fitness, medicines, and test conditions.',
    reason: 'The heart rate reached during exercise is interpreted with age, fitness, medicines, and test conditions.',
    source: 'American Heart Association: Target heart rates',
    sourceUrl: 'https://www.heart.org/en/healthy-living/fitness/fitness-basics/target-heart-rates',
  },
  exang: {
    label: 'Chest pain during exercise',
    clinicalRange: 'Yes or No, as recorded.',
    clinicalNote: 'This is a symptom question, not a diagnosis.',
    reason: 'Chest discomfort during activity can happen when the heart needs more oxygen-rich blood.',
    source: 'American Heart Association: Angina',
    sourceUrl: 'https://www.heart.org/en/health-topics/heart-attack/angina-chest-pain',
  },
  oldpeak: {
    label: 'ECG change during exercise',
    clinicalRange: 'No single patient-use cutoff.',
    clinicalNote: 'A clinician interprets this ECG measurement with the test method and other findings.',
    reason: 'Changes in the ECG during exercise can help clinicians assess how the heart responds to activity.',
    source: 'American Heart Association: Stress tests',
    sourceUrl: 'https://www.heart.org/en/health-topics/heart-attack/diagnosing-a-heart-attack/stress-tests',
  },
  slope: {
    label: 'ECG pattern during exercise',
    clinicalRange: 'No single stand-alone normal range.',
    clinicalNote: 'Choose the direction recorded in the source data; it is interpreted alongside symptoms and other test findings.',
    reason: 'How the ECG line changes during exercise is interpreted alongside symptoms and other test findings.',
    source: 'American Heart Association: Stress tests',
    sourceUrl: 'https://www.heart.org/en/health-topics/heart-attack/diagnosing-a-heart-attack/stress-tests',
  },
};

export function patientOutput(output) {
  if (typeof output === 'string') return output;
  return Number(output) === 1
    ? 'The model found patterns that may be associated with a possibility of heart disease.'
    : 'The model did not find patterns associated with a possibility of heart disease.';
}

export function patientInputValue(feature, value) {
  const choices = {
    sex: { 0: 'Female', 1: 'Male' },
    cp: {
      0: 'Typical angina',
      1: 'Atypical angina',
      2: 'Non-anginal chest pain',
      3: 'No chest pain reported',
    },
    fbs: { 0: 'No', 1: 'Yes' },
    restecg: {
      0: 'Normal',
      1: 'ST-T wave abnormality',
      2: 'Possible left ventricular hypertrophy pattern',
    },
    exang: { 0: 'No', 1: 'Yes' },
    slope: { 0: 'Upsloping', 1: 'Flat', 2: 'Downsloping' },
  };
  const fieldChoices = choices[feature];
  if (fieldChoices) return fieldChoices[value] ?? String(value);
  const units = {
    age: 'years',
    trestbps: 'mmHg',
    chol: 'mg/dL',
    thalach: 'beats per minute',
  };
  return units[feature] ? `${value} ${units[feature]}` : String(value);
}

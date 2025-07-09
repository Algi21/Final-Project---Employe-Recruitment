# Final-Project---Employee-Eligibility-Prediction

#Project Overview
- Problems:<br>
Perusahaan menghadapi kesulitan dalam menilai kelayakan kandidat secara objektif dan efisien akibat tingginya volume pelamar dan kompleksitas atribut seperti pendidikan, pengalaman, keterampilan, skor tes, dan referensi. Untuk mengatasi tantangan ini, diperlukan model prediktif berbasis data yang mampu mengidentifikasi faktor-faktor utama yang memengaruhi kelayakan kandidat. Model ini bertujuan meningkatkan efisiensi proses rekrutmen, mengurangi bias subjektif, dan mendukung pengambilan keputusan yang lebih akurat dengan memprediksi kelayakan kandidat secara otomatis berdasarkan data yang tersedia.

- Goals :
  Model ini bertujuan mengidentifikasi faktor utama yang memengaruhi kelayakan kandidat dan membantu menilai kandidat di masa depan. Dengan menganalisis data seperti pendidikan, pengalaman, keterampilan, rujukan, dan skor tes, model ini diharapkan dapat membuat proses perekrutan lebih efisien dan mendukung keputusan yang lebih tepat.

- Objective:
  - Mengidentifikasi Faktor Penentu dalam Kelayakan Rekrutmen
  - Mengembangkan Model Prediktif Kelayakan Kandidat
  - Meningkatkan Efisiensi Proses Rekrutmen
  - Mendukung Keputusan Berbasis Data (Data-Driven Hiring)

- Benefits :
  - Penghematan Waktu dan Biaya Rekrutmen
  - Peningkatan Kualitas Rekrutmen
  - Pengambilan Keputusan Lebih Objektif
  - Optimasi Sumber Daya HR : Peningkatan Retensi Karyawan

- Risiko Kualitas Data
Deskripsi:
Data yang digunakan dalam proyek ini berpotensi memiliki ketidakseimbangan distribusi pada label target (misalnya jumlah kandidat yang "layak" jauh lebih sedikit dibandingkan "tidak layak"). Ketidakseimbangan ini dapat menyebabkan model bias dan gagal mengenali pola penting dari kelompok minoritas.
Mitigasi:
Menerapkan teknik penyeimbangan data seperti SMOTE (Synthetic Minority Over-sampling Technique) atau penyesuaian class weight dalam algoritma, serta memantau metrik evaluasi seperti Recall dan F1-Score, bukan hanya akurasi.

- Risiko Ketepatan Model
Deskripsi:
Model yang dibangun mungkin mengalami overfitting, yaitu terlalu menyesuaikan diri dengan data pelatihan sehingga tidak mampu memberikan prediksi yang baik pada data baru (real-world data). Hal ini bisa menyebabkan akurasi tinggi saat pelatihan, tetapi buruk dalam implementasi.
Mitigasi:
Menerapkan teknik validasi silang (cross-validation), menggunakan metode regularisasi (seperti L1/L2), serta melakukan evaluasi model menggunakan data uji yang terpisah untuk memastikan generalisasi model yang baik.

- Risiko SDM dan Timeline
Deskripsi:
Keterbatasan jumlah anggota tim proyek dan tekanan waktu penyelesaian dapat menyebabkan proses pengolahan data, pengembangan model, hingga validasi akhir dilakukan secara terburu-buru. Hal ini berisiko menurunkan kualitas hasil analisis dan prediksi.
Mitigasi:
Membuat jadwal proyek yang terstruktur dan realistis, mengalokasikan tugas secara proporsional sesuai keahlian anggota tim, serta melakukan review berkala terhadap progres dan hambatan untuk menjaga kualitas dan ketepatan waktu pelaksanaan.

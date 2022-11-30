# Инструкция по запуску
1. Перейдите в папку document-ocr-service.
2. Затем необходимо установить все необходимые библиотеки из requirements.txt
  - Для этого нужно выполнить команду pip3 install -r requirements.txt
  - (ВАЖНО): Для работы приложения необходима БД PostgreSQL. Если её нет, необходимо ее установить, выполнив необходимые инструкции под вашу ОС: https://www.postgresql.org/download/linux/ 

3. Затем необходимо создать базу данных и загрузить все данные в нее. Это делается при помощи следующих команд:
  - Загрузка данных в БД для диплома: python3 src/document_ocr_diploma/updating_diplom/optimize_diplom/searchBD_diplom/data_diplom/to_bd_dp.py
  - Загрузка данных в БД для паспорта: python3 src/document_ocr_passport/modify_passport/image_updating_passport/findBD_passport/data_passport/to_bd.py

## Для запуска OCR для диплома:
1. В папке media/diploma_photo нужно загрузить нужные вам сканы дипломов. Они должны быть названы diplom{номер_фото}.jpg.
2. В файле src/document_ocr_diploma/main_diplom.py в переменную diplom_names записать [{номер_фото}].
3. Запуск выполнять из папки document-ocr-service.
4. Выполнить команду python3 src/document_ocr_diploma/main_diplom.py

## Для запуска OCR для паспорта:
1. В папке media/passport_photo нужно загрузить нужные вам сканы паспортов. Они должны быть названы orig{номер_фото}.jpg.
2. В файле src/document_ocr_passport/main_passport.py в переменную passport_names записать [{номер_фото}].
3. Запуск выполнять из папки document-ocr-service.
4. Выполнить команду python3 src/document_ocr_passport/main_passport.py

## Результаты
Результаты распознавания текстовой информации будут записаны в папку info, в папку time - время выполнения программы.
Для диплома они находятся в results/diploma_results, для паспорта - results/passport_results.

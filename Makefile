CONTAINER_NAME = energy_mlops_container
FULL_PIPELINE = run_full_pipeline.py
# Cache kontrolü için processed klasörünün varlığına bakıyoruz
DATA_DIR = data/processed

.PHONY: up down deploy test all pipeline wait-prefect

# 1️⃣ Ana komut: Her şeyi sırayla ve güvenle yapar
all: up wait-prefect pipeline deploy test

# 2️⃣ Docker Altyapısı
up:
	docker compose up -d --build
	@echo "Konteynerler ayağa kalktı. Servislerin ısınması bekleniyor..."

# 3️⃣ Prefect API Bekleme Mekanizması (Race Condition Önleyici)
wait-prefect:
	@echo "⏳ Prefect API'nin hazır olması bekleniyor (http://localhost:4200/api/health)..."
	@until curl -s http://localhost:4200/api/health > /dev/null; do \
		echo "... API henüz hazır değil, bekleniyor ..."; \
		sleep 3; \
	done
	@echo "✅ Prefect API hazır!"

# 4️⃣ Pipeline Tetikleyici
pipeline:
	@echo "Veri kontrol ediliyor..."
	@if [ ! -d "$(DATA_DIR)" ] || [ -z "$$(ls -A $(DATA_DIR) 2>/dev/null)" ]; then \
		echo "🔄 Veri eksik veya klasör boş. Full pipeline başlatılıyor..."; \
		docker exec $(CONTAINER_NAME) python3 $(FULL_PIPELINE); \
	else \
		echo "✅ İşlenmiş veri mevcut. Pipeline adımı atlanıyor (Cache)."; \
	fi

# 5️⃣ Prefect Deployment
deploy:
	docker exec -it $(CONTAINER_NAME) prefect deploy src/training/evaluate_and_promote_flow.py:evaluate_and_promote \
		--name daily-evaluate-promote --pool default-agent-pool --cron "0 0 * * *"

# 6️⃣ Temizlik
down:
	docker compose down -v

# 7️⃣ Test
test:
	@echo "Servisler kontrol ediliyor..."
	@curl -s http://localhost:4200/api/health || echo "Prefect beklemede..."
	@curl -s http://localhost:5000 || echo "MLflow beklemede..."
	@curl -s http://localhost:8000/health || echo "API beklemede..."
BEGIN TRANSACTION;
DROP TABLE IF EXISTS "voice_notes";
CREATE TABLE IF NOT EXISTS "voice_notes" (
	"id"	INTEGER,
	"filename"	TEXT,
	"file_path"	TEXT,
	"date_recorded"	TIMESTAMP,
	"date_processed"	TIMESTAMP,
	"transcription"	TEXT,
	"summary"	TEXT,
	"calls_to_action"	TEXT,
	"tone"	TEXT,
	"people_mentioned"	TEXT,
	"tags"	TEXT,
	"subject"	TEXT,
	"ai_provider"	TEXT,
	"ai_model"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT)
);
COMMIT;

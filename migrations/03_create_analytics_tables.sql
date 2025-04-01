-- Migration: 03_create_analytics_tables
-- Description: Creates tables for analytics data storage
-- Author: Claude API

-- Analytics Events Table
CREATE TABLE IF NOT EXISTS analytics_events (
    id VARCHAR(36) PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255),
    user_id VARCHAR(36),
    timestamp DATETIME NOT NULL,
    duration_ms FLOAT,
    status_code INT,
    error TEXT,
    metadata JSON,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_event_type (event_type),
    INDEX idx_user_id (user_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_endpoint (endpoint),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Webhook Subscriptions Table
CREATE TABLE IF NOT EXISTS webhook_subscriptions (
    id VARCHAR(36) PRIMARY KEY,
    url VARCHAR(2048) NOT NULL,
    owner_id VARCHAR(36) NOT NULL,
    secret VARCHAR(255),
    description TEXT,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_triggered DATETIME,
    success_count INT NOT NULL DEFAULT 0,
    failure_count INT NOT NULL DEFAULT 0,
    INDEX idx_owner_id (owner_id),
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Webhook Event Types Table (Many-to-Many)
CREATE TABLE IF NOT EXISTS webhook_event_types (
    webhook_id VARCHAR(36) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    PRIMARY KEY (webhook_id, event_type),
    FOREIGN KEY (webhook_id) REFERENCES webhook_subscriptions(id) ON DELETE CASCADE,
    INDEX idx_event_type (event_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Webhook Delivery Logs
CREATE TABLE IF NOT EXISTS webhook_delivery_logs (
    id VARCHAR(36) PRIMARY KEY,
    webhook_id VARCHAR(36) NOT NULL,
    event_id VARCHAR(36) NOT NULL,
    status_code INT,
    success BOOLEAN NOT NULL,
    attempt_count INT NOT NULL DEFAULT 1,
    response_body TEXT,
    error TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (webhook_id) REFERENCES webhook_subscriptions(id) ON DELETE CASCADE,
    INDEX idx_webhook_id (webhook_id),
    INDEX idx_event_id (event_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Analytics Aggregated Metrics Table (Daily)
CREATE TABLE IF NOT EXISTS analytics_daily_metrics (
    date DATE PRIMARY KEY,
    total_api_calls INT NOT NULL DEFAULT 0,
    total_users INT NOT NULL DEFAULT 0,
    pdf_processing_count INT NOT NULL DEFAULT 0,
    search_count INT NOT NULL DEFAULT 0,
    question_count INT NOT NULL DEFAULT 0,
    error_count INT NOT NULL DEFAULT 0,
    avg_response_time FLOAT NOT NULL DEFAULT 0,
    p95_response_time FLOAT NOT NULL DEFAULT 0,
    max_response_time FLOAT NOT NULL DEFAULT 0,
    endpoints JSON,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Add permissions for analytics
INSERT IGNORE INTO permissions (name, description) VALUES 
('view_analytics', 'Can view analytics data'),
('manage_webhooks', 'Can manage webhook subscriptions'),
('view_all_analytics', 'Can view analytics data for all users');

-- Add analytics permissions to admin role
INSERT IGNORE INTO role_permissions (role_id, permission_id)
SELECT 
    (SELECT id FROM roles WHERE name = 'admin'),
    id
FROM 
    permissions 
WHERE 
    name IN ('view_analytics', 'manage_webhooks', 'view_all_analytics');

-- Add basic analytics permissions to user role
INSERT IGNORE INTO role_permissions (role_id, permission_id)
SELECT 
    (SELECT id FROM roles WHERE name = 'user'),
    id
FROM 
    permissions 
WHERE 
    name IN ('view_analytics', 'manage_webhooks'); 
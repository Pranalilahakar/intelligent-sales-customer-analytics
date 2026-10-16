
SELECT
    p.stock_code,
    p.description,
    ROUND(SUM(oi.revenue), 2) AS total_revenue,
    SUM(oi.quantity) AS units_sold,
    COUNT(DISTINCT oi.invoice_no) AS orders_count
FROM order_items oi
JOIN products p
    ON oi.stock_code = p.stock_code
GROUP BY p.stock_code, p.description
ORDER BY total_revenue DESC
LIMIT 10;

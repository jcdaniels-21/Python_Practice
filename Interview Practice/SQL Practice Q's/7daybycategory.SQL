SELECT ITEMS.item_category AS "Category", SUM(CASE WHEN ORDERS.order_quantity IS NULL THEN 0 ELSE ORDERS.order_quantity END) AS "Units" FROM
ORDERS LEFT JOIN ITEMS ON ORDERS.item_id = ITEMS.item_id
WHERE ORDERS.order_datetime BETWEEN CURRENT_DATE-7 AND CURRENT_DATE OR ORDERS.order_quantity IS NULL
GROUP BY ITEMS.item_category;